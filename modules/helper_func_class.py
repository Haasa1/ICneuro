# this lib, together with the preprocess_mice_trials script
# is used to process the raw excel files that have the tracking
# of the mouse in the arena looking for the hidden food
#
# the experiments were originally performed by Kelly Xu from Len Maler lab
#
# the original excel files are not going to be included in this repository
#
# instead, we provide only the processed files (in .mat format)
# that contain all the relevant information for the trajectories
# extracted from the excel files...

import os
import math
import copy
import numpy
import operator
import warnings
import collections
import scipy.io
import scipy.stats
import scipy.sparse
import scipy.spatial
import scipy.optimize
import scipy.interpolate
import modules.io as io
from enum import Enum
from numba import jit


#@jit(nopython=True,parallel=True, fastmath=True)
def intertwin_vectors(r1,r2,*rn,copy_data=True,axis=0):
    """
    merges r1 and r2 in r,
    such that r[0::2] = r1
    and       r[1::2] = r2
    """
    if copy_data:
        get_value = lambda x: copy.deepcopy(x)
    else:
        get_value = lambda x: x
    rn                 = (r1,r2) + rn
    use_rows_intertwin = True
    if type(r1) is list:
        assert all((type(r) is list) and (len(r) == len(r1)) for r in rn),"all input r must be lists because r1 is a list; and all must have the same length"
        r = [None for _ in range(len(rn)*len(r1)) ]
    else:
        if r1.ndim == 1:
            assert all((r.ndim == 1) and (r.size == r1.size) for r in rn),"all input r must be the same ndim and size"
            r = numpy.zeros(r1.size * len(rn),dtype=r1.dtype)
        elif r1.ndim == 2:
            assert all((r.ndim == 2) and (r.shape == r1.shape) for r in rn),"r1 and r2 must be the same ndim and shape"
            assert any(axis == n for n in (0,1)), "axis must be 0 (for row intertwining) or 1 (for column intertwining)"
            if axis == 0:
                r = numpy.zeros((r1.shape[0]*len(rn),r1.shape[1]),dtype=r1.dtype)
            else:
                r = numpy.zeros((r1.shape[0],r1.shape[1]*len(rn)),dtype=r1.dtype)
                use_rows_intertwin = False
        else:
            raise ValueError('this function is not defined for ndim > 2')
    if use_rows_intertwin:
        N = len(rn)
        for n in range(N):
            r[n::N] = get_value(rn[n])
    else:
        N = len(rn)
        for n in range(N):
            r[:,n::N] = get_value(rn[n])
    return r

#@jit(nopython=True,parallel=True, fastmath=True)
def to_cartesian(r,theta=None):
    """
    converts the r,theta pairs, r[t],theta[t], into x,y coordinates stored in a 2d ndarray
    xy[:,:2], such that xy[t,0] -> x[t] and xy[t,1] -> y[t]

    if theta is not set, r must be a Nx2 ndarray, 1st col -> r, 2nd col -> theta
    """
    if type(theta) is type(None):
        assert ((type(r) is numpy.ndarray) and (r.ndim == 2) and (r.shape[1] == 2)),"if theta is not set, r must be a Nx2 ndarray, 1st col -> r, 2nd col -> theta"
        theta = r[:,1]
        r     = r[:,0]
    else:
        assert ((type(r) is numpy.ndarray) and (r.ndim == 1)), "r must be a flat numpy.ndarray"
        assert ((type(theta) is numpy.ndarray) and (theta.ndim == 1)), "theta must be a flat numpy.ndarray"
    return numpy.column_stack((r*numpy.cos(theta),r*numpy.sin(theta)))

#@jit(nopython=True,parallel=True, fastmath=True)
def to_polar(r):
    """
    converts the vectors in r to polar coordinates, norm(r), theta

    each row in r is a vector to be converted to polar
    r[k,0] -> x[k]
    r[k,1] -> y[k]
    is converted to
    r_norm[k],theta[k]
    """
    assert ((type(r) is numpy.ndarray) and (r.ndim == 2)), "r must be a 2d numpy.ndarray"
    return numpy.linalg.norm(r,axis=1),numpy.arctan2(r[:,1],r[:,0])

#@jit(nopython=True,parallel=True, fastmath=True)
def derivative(x,f,axis=0,interpolate=False,epsilon=1.0e-10):
    """
    estimates f'(x), where f' = df/dx
    
    either x.shape == f.shape
          or x is flat and x.size == f.shape[axis]
          or x and f are flat and f.size==x.size

    axis :: vector axis over which to consider f(x); if axis==0, the function is defined over the rows (each column of f is a different function)
                                                        axis==1, the function is defined over  the columns (each row of f is a different function)
    interpolate :: if true, interpolates f over x with a cubic polynomial
                    the downside is that the derivative is always a square function
                    the advantage is that it may get very accurate for polynomial functions f(x)
    epsilon :: only useful if interpolate==True; precision of the derivative in the defintion f'(x) = (f(x+epsilon) - f(x))/epsilon

    returns a vector the same size as f
    """
    errMsg = 'x and f must be flat numpy arrays, such that either: x.shape == f.shape; or x is flat and x.size == f.shape[axis]; or x and f are flat and f.size==x.size'
    if not((type(x) is numpy.ndarray) and (type(f) is numpy.ndarray)):
        raise ValueError(errMsg)
    if x.shape != f.shape:
        if (x.ndim > 1) and (f.ndim == 1): # if f is flat, x must be flat
            raise ValueError(errMsg)
        if (x.ndim == 1) and (f.ndim == 1):
            if x.size != f.size: # both are  flat, but sizes dont match
                raise ValueError(errMsg)
        if (x.ndim == 1) and (x.size != f.shape[axis]): # x is flat, but its size doesnt match the appropriate size in f
            raise ValueError(errMsg)
        if (x.ndim > 2) or (f.ndim > 2):
            raise ValueError('x and f must be 2d, 1d or flat arrays; ' + errMsg)
        if (x.ndim > 1) and (f.ndim > 1):
            if (x.ndim != f.ndim) or (x.shape[axis] != f.shape[axis]): # they are both not flat, but shapes dont match
                raise ValueError(errMsg)
    if f.ndim > 1:
        get_x = lambda xx,kk: xx if xx.ndim == 1 else (xx[:,kk] if axis == 0 else xx[kk,:])
        get_f = lambda ff,kk: ff[:,kk] if axis == 0 else ff[kk,:]
        N = f.shape[1-axis] # number of functions
        T = f.shape[axis]
        fp = numpy.empty(shape=(N,T))
        for k in range(N):
            fp[k,:] = derivative(get_x(x,k),get_f(f,k),interpolate=interpolate,epsilon=epsilon)
        return fp if axis == 1 else fp.T
    else:
        if interpolate:
            ff   = scipy.interpolate.interp1d(x,f,kind='cubic',copy=False)
            dfdx = lambda x: (ff(x+epsilon) - ff(x))/epsilon
            return numpy.append(dfdx(x[:-1]), dfdx(x[-1]-epsilon))
            # we append fp[-1] because we use x[:-1] in the approx_fprime call
            #return numpy.append(scipy.optimize.approx_fprime(x[:-1],lambda x: ff([x])[0] if numpy.isscalar(x) else ff(x)[0],epsilon=epsilon*numpy.ones(x.size-1)),fp[-1])
        else:
            return numpy.gradient(f,x,edge_order=2)

def mean_std_err_minmax(x,axis=0,return_as_struct=False):
    if axis > 1:
        raise ValueError('function not defined for multidim arrays with ndim>2')
    avg = nanmean(x,axis=axis)
    sd  =  nanstd(x,axis=axis)
    se  = nanserr(x,axis=axis)
    m   =  nanmin(x,axis=axis)
    M   =  nanmax(x,axis=axis)
    if return_as_struct:
        return structtype(avg=avg,sd=sd,se=se,min=m,max=M)
    else:
        return avg,sd,se,m,M

def cohen_d(x,y,axis=0):
    """
    axis -> observations in x and y
    if axis == 0: col(x) must equal col(y): observations are in rows, so compares each column k of x with each column k of y
    if axis == 1: row(x) must equal row(y): observations are in columns, so compares each row k of x with each row k of y
    """
    if not(axis in [0,1]):
        raise ValueError('cohen_d ::: axis must be 0 or 1')
    x = x if _is_numpy_array(x) else numpy.array(x)
    y = y if _is_numpy_array(y) else numpy.array(y)
    otheraxis = 1 - axis
    if (x.ndim > 1):
        if (y.ndim != x.ndim):
            raise ValueError('cohen_d ::: ndim must match in x and y')
        if x.shape[otheraxis] != y.shape[otheraxis]:
            raise ValueError('cohen_d ::: if axis == 0, the number of columsn in x and y must match; for x == 1, the number of rows must match')
    nx  = x.shape[axis] if x.ndim > 1 else x.size
    ny  = y.shape[axis] if y.ndim > 1 else y.size
    dof = nx + ny - 2
    return (nanmean(x,axis=axis) - nanmean(y,axis=axis)) / numpy.sqrt(((nx-1)*nanstd(x, ddof=1, axis=axis) ** 2 + (ny-1)*nanstd(y, ddof=1, axis=axis) ** 2) / dof)

def minmax(x):
    return nanmin(x),nanmax(x)

def stderr_sum(*se):
    """
    calculates the std error of a sum or subtraction of variables
    (the sum of errors)

    se[k] -> standard error of the k-th variable...
    all se[k] must be of the same size (i.e., 1d numpy.ndarray of size N)

    ref: https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
    """
    return numpy.sum(numpy.array([numpy.array(s).flatten() for s in se]),axis=0)

def stddev_sum(*sd):
    """
    calculates the std deviation of a sum or subtraction of variables
    (the sqrt of the sum of squared std devs)

    sd[k] -> standard deviation of the k-th variable...
    all sd[k] must be of the same size (i.e., 1d numpy.ndarray of size N)

    ref: https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
    """
    return numpy.sqrt(numpy.sum(numpy.array([numpy.array(s).flatten()**2 for s in sd]),axis=0))

def minmax_sum(Z,X_minmax):
    """
    Z        -> X[1] + X[2] + ... (+ or -)
    X_minmax -> min (col 0) max (col 1) of each X variable: X_minmax[0] -> (min,max) of X[0]

    this function converts the interval abs(max - min) as if it were a SE DeltaX, and then calculates min = Z - abs(max-min)/2; max = Z + abs(max - min)/2
    """
    dX       = [ numpy.abs(numpy.diff(mm,axis=1).flatten()) for mm in X_minmax ]
    dZ       = stderr_sum(*dX)/2.0
    Z_minmax = numpy.array([Z - dZ,Z + dZ]).T
    return Z,Z_minmax

def stderr_mult(Z,X,X_SE):
    """
    calculates the std error of the product or division of variables in X

    Z is an 1d numpy.ndarray; all X[k] and X_SE[k] must match Z in type, size and shape

    Z    -> resulting of the multiplication or division of X variables; e.g. X[0]*X[1]*...*X[N] = Z
    X    -> tuple; each entry is a variable; e.g., X[0]*X[1]*...*X[N] = Z for multiplication
    X_SE -> tuple; each entry is the stderr of the corresponding X variable: X_SE[k] is the stderr of X[k]

    ref: https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
    """
    X    = numpy.array([numpy.array(x).flatten() for x in X   ])
    X_SE = numpy.array([numpy.array(s).flatten() for s in X_SE])
    Z    = Z if _is_numpy_array(Z) else numpy.array(Z)
    Z_SE = Z * ( numpy.sum( X_SE/X, axis=0 ) )
    return Z,Z_SE

def stddev_mult(Z,X,X_SD):
    """
    calculates the std deviation of the product or division of variables in X

    Z is an 1d numpy.ndarray; all X[k] and X_SD[k] must match Z in type, size and shape

    Z    -> resulting of the multiplication or division of X variables; e.g. X[0]*X[1]*...*X[N] = Z
    X    -> tuple; each entry is a variable; e.g., X[0]*X[1]*...*X[N] = Z for multiplication
    X_SD -> tuple; each entry is the stderr of the corresponding X variable: X_SD[k] is the stderr of X[k]

    ref: https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
    """
    X    = numpy.array([numpy.array(x).flatten() for x in X   ])
    X_SD = numpy.array([numpy.array(s).flatten() for s in X_SD])
    Z    = Z if _is_numpy_array(Z) else numpy.array(Z)
    Z_SE = Z * numpy.sqrt( numpy.sum( (X_SD/X)**2, axis=0 ) )
    return Z,Z_SE

def minmax_mult(Z,X,X_minmax):
    """
    Z        -> X[1] * X[2] * ... (* or /)
    X        -> tuple; each entry is a variable; e.g., X[0]*X[1]*...*X[N] = Z for multiplication
    X_minmax -> min (col 0) max (col 1) of each X variable: X_minmax[0] -> (min,max) of X[0]

    this function converts the interval abs(max - min) as if it were a SE DeltaX, and then calculates min = Z - abs(max-min)/2; max = Z + abs(max - min)/2
    """
    dX       = [ numpy.abs(numpy.diff(mm,axis=1).flatten()) for mm in X_minmax ]
    dZ       = stderr_mult(Z,X,dX)[1]/2.0
    Z_minmax = numpy.array([Z - dZ,Z + dZ]).T
    return Z,Z_minmax

def avg_of_avg(x_avg,x_std,x_err,x_min=None,x_max=None,axis=None):
    """
    returns the average of averages

    the average std is the sqrt of the average variance

    returns:
        * avg of x_avg
        * avg of x_std (i.e. sqrt of the average variance)
        * avg of x_err (i.e. the avg stddev divided by the sqrt of the number of elements in x_avg)
        * if x_min is passed, then returns the min of min
        * if x_max is passed, then returns the max of max
    """
    x_avg = numpy.asarray(x_avg) if not _is_numpy_array(x_avg) else x_avg
    x_std = numpy.asarray(x_std) if not _is_numpy_array(x_std) else x_std
    x_err = numpy.asarray(x_err) if not _is_numpy_array(x_err) else x_err
    s = numpy.sqrt(nanmean(x_std**2,axis=axis))
    n = _get_number_of_valid_elements(x_avg,axis=axis)
    result = nanmean(x_avg,axis=axis),s,s/numpy.sqrt(n)
    if exists(x_min):
        x_min = numpy.asarray(x_min) if not _is_numpy_array(x_min) else x_min
        result += (nanmin(x_min,axis=axis),)
    if exists(x_max):
        x_max = numpy.asarray(x_max) if not _is_numpy_array(x_max) else x_max
        result += (nanmax(x_max,axis=axis),)
    return result # x_avg,x_std,x_err,[[x_min | optional ]],[[x_max | optional]]

def _get_number_of_valid_elements(x,axis=None):
    """"""
    if not _is_numpy_masked_array(x):
        x = numpy.ma.masked_invalid(x)
    return numpy.expand_dims(x.count(axis=axis),axis) if exists(axis) else x.count()

def _is_numpy_masked_array(x):
    return type(x) is numpy.ma.core.MaskedArray #('numpy' in str(type(x))) and ('MaskedArray' in str(type(x)))

def _is_numpy_array(x):
    return type(x) is numpy.ndarray

def exists(x):
    return not(type(x) is type(None))

#@jit(nopython=True,parallel=True, fastmath=True)
def avg_angle_from_cos(c,axis=0,in_radians=False,return_minmax=False):
    """
    c is a matrix of cosine values
    if c is a list, then repeats this function for every entry in the list

    this function averages c over axis
    and returns the angles and the angles' std dev

    in_radians :: angles are returned in radians (degrees otherwise)
    """
    to_deg = 1.0
    if not in_radians:
        to_deg = 180.0 / numpy.pi
    if type(c) is list:
        angle_avg = get_empty_list(len(c))
        angle_std = get_empty_list(len(c))
        angle_err = get_empty_list(len(c))
        not_nan_count = get_empty_list(len(c))
        if return_minmax:
            min_angle = get_empty_list(len(c))
            max_angle = get_empty_list(len(c))
        for k,cc in enumerate(c):
            if return_minmax:
                angle_avg[k],angle_std[k],angle_err[k],not_nan_count[k],min_angle[k],max_angle[k] = avg_angle_from_cos(cc,axis=axis,in_radians=in_radians,return_minmax=return_minmax)
            else:
                angle_avg[k],angle_std[k],angle_err[k],not_nan_count[k] = avg_angle_from_cos(cc,axis=axis,in_radians=in_radians,return_minmax=return_minmax)
    else:
        if not _is_numpy_array(c):
            c = numpy.asarray(c)
        c[numpy.abs(c)>1.0] = numpy.sign(c[numpy.abs(c)>1.0]).astype(float)
        c_avg = nanmean(c,axis=axis) # calculating the avg over mice (mavg) for each learning stage
        if not numpy.isscalar(c_avg):
            c_avg[numpy.abs(c_avg)>1.0] = numpy.sign(c_avg[numpy.abs(c_avg)>1.0]).astype(float)
        c_std = nanstd(c,axis=axis) # calculating the avg over mice (mavg) for each learning stage
        angle_avg = numpy.arccos(c_avg)*to_deg
        angle_std = to_deg*numpy.sqrt((1.0/numpy.abs(1.0-c_std**2.0))) * c_std # error propagation from the cosine to the angle ( arccos(deviation) ), d(arccos x)/dx = -1/sqrt(1-x**2)
        sqrt_of_n = numpy.sqrt(numpy.sum((~numpy.ma.masked_invalid(c).mask).astype(float),axis=axis))
        angle_err = angle_std / sqrt_of_n #numpy.sqrt(c.shape[axis])
        if return_minmax:
            min_angle = nanmin(numpy.arccos(c)*to_deg,axis=axis)
            max_angle = nanmax(numpy.arccos(c)*to_deg,axis=axis)
        not_nan_count = numpy.sum(numpy.logical_not(numpy.isnan(c)).astype(numpy.int64),axis=axis)
    if return_minmax:
        return angle_avg,angle_std,angle_err,not_nan_count,min_angle,max_angle
    else:
        return angle_avg,angle_std,angle_err,not_nan_count

def cos_uv(u,v,axis=1):
    """
    returns the cosine of the angle between each vector in u (ndarray 1 vector per row) and v, or each vector in u and each vector in v if v is ndarray (1 vector per row)
    """
    if axis == 0:
        if u.ndim > 1:
            u = u.T
        if v.ndim > 1:
            v = v.T
    if v.ndim > 1:
        if u.ndim > 1:
            c = numpy.fromiter((numpy.dot(uu,vv) for uu,vv in zip(u,v)),dtype=float)/(numpy.linalg.norm(u,axis=axis)*numpy.linalg.norm(v,axis=axis))
        else:
            c = numpy.fromiter((numpy.dot(u,vv) for vv in v),dtype=float)/(numpy.linalg.norm(u)*numpy.linalg.norm(v,axis=axis))
    else:
        if u.ndim > 1:
            c = numpy.fromiter((numpy.dot(uu,v) for uu in u),dtype=float)/(numpy.linalg.norm(u,axis=axis)*numpy.linalg.norm(v))
        else:
            c = numpy.dot(u,v)/(numpy.linalg.norm(u)*numpy.linalg.norm(v))
    if not numpy.isscalar(c):
        c[numpy.abs(c) > 1.0] = numpy.round(c[numpy.abs(c) > 1.0])
    return c

def angle_uv(u,v,axis=1):
    """
    returns the angle between each vector in u (ndarray 1 vector per row) and v, or each vector in u and each vector in v if v is ndarray (1 vector per row)

    CCW angles (relative to v) are positive [0,180], CW are negative [-180,0]
    """
    sign = 2.0*numpy.asarray(is_to_the_right(u,v,axis),dtype=float)-1.0 # 1 if u is to the left of v; -1 if u is to the right of v
    return sign*numpy.arccos(cos_uv(u,v,axis=axis))

def is_to_the_right(u,v,axis=1):
    """
    returns True if v is to the right of u
    u and v are two vectors on the x,y plane
    u may be a list of vectors
    axis == 1 compares each row of u to v; axis == 0 compares each col of u to v
    """
    if axis == 0:
        if u.ndim > 1:
            u = u.T
        if v.ndim > 1:
            v = v.T
    if u.ndim > 1:
        if v.ndim > 1:
            return numpy.fromiter((numpy.dot(uu[:2],[-vv[1],vv[0]])>0.0 for uu,vv in zip(u,v)),dtype=bool) # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
        else:
            return numpy.dot(u[:,:2],[-v[1],v[0]])>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
    else:
        if v.ndim > 1:
            return numpy.fromiter((numpy.dot(u[:2],[-vv[1],vv[0]])>0.0 for vv in v),dtype=bool) # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
        else:
            return numpy.dot(u[:2],[-v[1],v[0]])>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v


# #@jit(nopython=True,parallel=True, fastmath=True)
# def _cos_uv(u,v,axis=1):
#     """
#     returns the cosine of the angle between each vector in u (ndarray 1 vector per row) and v, or each vector in u and each vector in v if v is ndarray (1 vector per row)
#     """
#     if axis == 0:
#         if u.ndim > 1:
#             u = u.T
#         if v.ndim > 1:
#             v = v.T
#     fix_value = lambda a: numpy.round(a) if numpy.abs(a) > 1.0 else a
#     if (u.shape[0] == 1) and (v.shape[0] == 1):
#         return fix_value(numpy.dot(u.flatten(),v.flatten())/(numpy.linalg.norm(u)*numpy.linalg.norm(v)))
#     if v.shape[0] > 1:
#         #assert (v.shape == u.shape),"u and v must be the same shape, 1 vector per row"
#         if u.shape[0] > 1:
#             c = numpy.zeros(u.shape[0])
#             for k in range(u.shape[0]):
#                 c[k] = fix_value(numpy.dot(u[k,:],v[k,:])/(numpy.linalg.norm(u[k,:])*numpy.linalg.norm(v[k,:])))
#         else:
#             c = numpy.zeros(v.shape[0])
#             for k in range(v.shape[0]):
#                 c[k] = fix_value(numpy.dot(u.flatten(),v[k,:])/(numpy.linalg.norm(u)*numpy.linalg.norm(v[k,:])))
#         #c = numpy.asarray([numpy.dot(uu,vv) for uu,vv in zip(u,v)])/(numpy.linalg.norm(u,axis=axis)*numpy.linalg.norm(v,axis=axis))
#     else:
#         c = numpy.zeros(u.shape[0])
#         for k in range(u.shape[0]):
#             c[k] = fix_value(numpy.dot(u[k,:],v.flatten())/(numpy.linalg.norm(u[k,:])*numpy.linalg.norm(v)))
#         #c = numpy.dot(u,v)/(numpy.linalg.norm(u,axis=axis)*numpy.linalg.norm(v))
#     #c[numpy.abs(c) > 1.0] = numpy.round(c[numpy.abs(c) > 1.0])
#     return c

# def cos_uv(u,v,axis=1):
#     """
#     returns the cosine of the angle between each vector in u (ndarray 1 vector per row) and v, or each vector in u and each vector in v if v is ndarray (1 vector per row)
#     """
#     u = u if u.ndim > 1 else u.reshape((1,u.size))
#     v = v if v.ndim > 1 else v.reshape((1,v.size))
#     return _cos_uv(u,v,axis)

# #@jit(nopython=True,parallel=True, fastmath=True)
# def _angle_uv(u,v,axis=1):
#     """
#     returns the angle between each vector in u (ndarray 1 vector per row) and v, or each vector in u and each vector in v if v is ndarray (1 vector per row)

#     CCW angles (relative to v) are positive [0,180], CW are negative [-180,0]
#     """
#     sign = 2.0*numpy.asarray(_is_to_the_right(u,v,axis),dtype=numpy.float64)-1.0 # 1 if u is to the left of v; -1 if u is to the right of v
#     return sign*numpy.arccos(_cos_uv(u,v,axis=axis))

# def angle_uv(u,v,axis=1):
#     """
#     returns the angle between each vector in u (ndarray 1 vector per row) and v, or each vector in u and each vector in v if v is ndarray (1 vector per row)

#     CCW angles (relative to v) are positive [0,180], CW are negative [-180,0]
#     """
#     u = u if u.ndim > 1 else u.reshape((1,u.size))
#     v = v if v.ndim > 1 else v.reshape((1,v.size))
#     return _angle_uv(u,v,axis)

# #@jit(nopython=True,parallel=True, fastmath=True)
# def _is_to_the_right(u,v,axis=1):
#     """
#      returns True if v is to the right of u
#      u and v are two vectors on the x,y plane
#      u may be a list of vectors
#      axis == 1 compares each row of u to v; axis == 0 compares each col of u to v
#     """
#     if axis == 0:
#         if u.ndim > 1:
#             u = u.T
#         if v.ndim > 1:
#             v = v.T
#     rotate = lambda x: numpy.dot(numpy.array([[-1.0,0.0],[0.0,1.0]]),x.flatten())
#     if u.shape[0] > 1:
#         r = numpy.zeros(u.shape[0],dtype=numpy.bool_)#[ False for _ in range(u.shape[0]) ]
#         if v.shape[0] > 1:
#             for k in range(u.shape[0]):
#                 r[k] = numpy.dot(u[k,:],rotate(v[k,:]).flatten())>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
#             #return numpy.asarray([numpy.dot(uu[:2],[-vv[1],vv[0]])>0.0 for uu,vv in zip(u,v)]) # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
#         else:
#             for k in range(u.shape[0]):
#                 r[k] = numpy.dot(u[k,:],rotate(v).flatten())>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
#             #return numpy.dot(u[:,:2],[-v[1],v[0]])>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
#     else:
#         if v.shape[0] > 1:
#             r = numpy.zeros(v.shape[0],dtype=numpy.bool_)#r = [ False for _ in range(v.shape[0]) ]
#             for k in range(v.shape[0]):
#                 r[k] = numpy.dot(u,rotate(v[k,:]).flatten())>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
#             #return numpy.asarray([numpy.dot(u[:2],[-vv[1],vv[0]])>0.0 for vv in v]) # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
#         else:
#             #r = [False]
#             return numpy.dot(u,rotate(v).flatten())>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v
#     return r #numpy.array(r,dtype=numpy.int64)

# def is_to_the_right(u,v,axis=1):
#     """
#      returns True if v is to the right of u
#      u and v are two vectors on the x,y plane
#      u may be a list of vectors
#      axis == 1 compares each row of u to v; axis == 0 compares each col of u to v
#     """
#     u = u if u.ndim > 1 else u.reshape((1,u.size))
#     v = v if v.ndim > 1 else v.reshape((1,v.size))
#     return _is_to_the_right(u,v,axis)

class RotateTransf:
    def __init__(self,p0=None,p1=None,rot_o=None,angle=None):
        """
        defines a 2d rotation from p0 to p1 (2d points)
        p0 -> a vector that, after applying this transform, will be colinear to p1 and pointing to the same side
        p1 -> reference vector towards which we want to rotate p0
        rot_o -> axis of rotation in (x,y) around which we want to rotate p0 (and all the other vectors with this transform)
        angle -> (radians) if angle is passed, then ignore p0 and p1, and creates a rotation for the specified angle
        """
        has_p0       = not(type(p0)    is type(None))
        has_p1       = not(type(p1)    is type(None))
        has_angle    = not(type(angle) is type(None))
        has_rot_o    = not(type(rot_o) is type(None))
        if has_p0:
            assert has_p1, "if p0 is set, you must set p1"
        if has_p1:
            assert has_p0, "if p1 is set, you must set p0"
        if has_angle:
            p0 = p0 if has_p0 else numpy.ones(3)
            p1 = p1 if has_p1 else -numpy.ones(3)
        self.rot_o   = numpy.asarray(rot_o).flatten()[:2] if has_rot_o else numpy.zeros(2)
        self._p0     = numpy.append(numpy.asarray(p0).flatten()[:2],0)# - self.rot_o
        self._p1     = numpy.append(numpy.asarray(p1).flatten()[:2],0)# - self.rot_o
        self._angle  = angle
        self.theta   = 0.0
        self._cos_th = 0.0
        self._sin_th = 0.0
        self.R       = numpy.zeros((3,3))
        self.CalcMatrix()
    def CalcMatrix(self,inv=1.0):
        # if inv is set, rotate to the opposite direction
        if self._angle is None:
            # we want to rotate vector p0 to align it to p1
            # we first calculate the angle between p0 and p1
            self.theta = self._angle_uv(self._p0,self._p1)
            # then, we need to figure out whether to rotate p0 to the right (CW) or to the left (CCW)
            # the matrix defined below rotates to the left (CCW) if theta > 0
            # and to the right (CW) if theta < 0
            # then, if p1 is to the right of p0, we need to change the sign of theta to be able to rotate to the right (CW)
            # because the function _angle_uv always returns a positive angle
            self.theta = -self.theta if self._is_to_the_right(self._p0,self._p1) else self.theta #changing the sign of theta as necessary
        else:
            self.theta = self._angle
        if inv != 0:
            self.theta = numpy.sign(float(inv)) * self.theta
        # consider the center of rotation at
        # rot_o = (x,y)
        # then the matrix below is the product between the translation matrix from rot_o towards (0,0):
        #      T' = [1,0,-x,
        #            0,1,-y,
        #            0,0, 1]
        # the rotation matrix
        #      R = [ cos(th) , -sin(th), 0,
        #            sin(th) ,  cos(th), 0,
        #               0    ,    0    , 1]
        # and the translation back from (0,0) towards rot_o
        #       T = [1,0,x,
        #            0,1,y,
        #            0,0,1]
        # yielding T R T' v = v_rotated
        # https://math.stackexchange.com/a/2093322/59039
        # the matrix below is calculated but not used!
        self.R = numpy.matmul(numpy.array([ [1,0,self.rot_o[0]],[0,1,self.rot_o[1]],[0,0,1] ],float),
                 numpy.matmul(numpy.array([ [numpy.cos(self.theta),-numpy.sin(self.theta),0],[numpy.sin(self.theta),numpy.cos(self.theta),0], [0,0,1] ],float),
                              numpy.array([ [1,0,-self.rot_o[0]],[0,1,-self.rot_o[1]],[0,0,1] ],float)))
        # however, for faster calculations, we can perform this product in advance (analytically)
        # and derive the transformation that yield the x component and the y component separately
        self._cos_th = numpy.cos(self.theta)
        self._sin_th = numpy.sin(self.theta)
    def _transf_x0(self,r):
        return self._cos_th*(r[0] - self.rot_o[0]) + self.rot_o[0] - self._sin_th * (r[1] - self.rot_o[1])
    def _transf_y0(self,r):
        return self._sin_th*(r[0] - self.rot_o[0]) + self.rot_o[1] + self._cos_th * (r[1] - self.rot_o[1])
    def _transf_x(self,r):
        return self._cos_th*(r[:,0] - self.rot_o[0]) + self.rot_o[0] - self._sin_th * (r[:,1] - self.rot_o[1])
    def _transf_y(self,r):
        return self._sin_th*(r[:,0] - self.rot_o[0]) + self.rot_o[1] + self._cos_th * (r[:,1] - self.rot_o[1])
    def Transf(self,X,inv=1.0):
        # if inv is set, rotate to the opposite direction
        if not(type(X) is numpy.ndarray):
            X = numpy.asarray(X)
        if (X.ndim > 2) or ((len(X.shape) > 1) and (X.shape[1] > 2)):
            raise ValueError('X can only be a 2d vector, or an N-by-2 matrix with one 2d vector per row')
        if inv < 0:
            self.CalcMatrix(inv) # we recalculate the matrix just in case we want to rotate in the inverse direction
        if X.ndim == 1:
            RX = numpy.array( ( self._transf_x0(X), self._transf_y0(X) ) )
        else:
            RX = numpy.column_stack( (self._transf_x(X), self._transf_y(X))  )
        if inv < 0:
            self.CalcMatrix(-inv) # if we had to recalculate, restore the original rotation object
        return RX
    def __call__(self,X,inv=1.0):
        # if inv is set, rotate to the opposite direction
        return self.Transf(X,inv)
    def _myacos(self,x):
        x = x if numpy.abs(x) <= 1.0 else numpy.round(x) # [numpy.abs(x) > 1.0] = numpy.round(x[numpy.abs(x) > 1.0])
        return numpy.arccos(x)
    def _angle_uv(self,u,v):
        return self._myacos(numpy.dot(u,v)/(numpy.linalg.norm(u)*numpy.linalg.norm(v)))
    def _is_to_the_right(self,u,v):
        # returns True if v is to the right of u
        # u and v are two vectors on the x,y plane
        # u may be a list of vectors (one vector per row)
        return numpy.dot(u[:2],[-v[1],v[0]])>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v

class LinearTransf:
    def __init__(self,X_lim,Y_lim):
        """ Creates a linear transformation from X to Y
        X_lim is the interval range of X and Y_lim is the interval range of Y
        """
        self.X_lim = numpy.asarray(X_lim).flatten()[0:2]
        self.Y_lim = numpy.asarray(Y_lim).flatten()[0:2]
        self.A = numpy.nan
        self.B = numpy.nan
        self.CalcCoeff()
    def CalcCoeff(self):
        self.A = numpy.diff(self.Y_lim) / numpy.diff(self.X_lim)
        self.B = self.Y_lim[0] - self.A * self.X_lim[0]
    def T(self,X):
        #return numpy.add(numpy.multiply(X,self.A),self.B)  # A*X + B
        Y = numpy.add(numpy.multiply(X,self.A),self.B)  # A*X + B
        return Y.flatten()[0] if Y.size == 1 else Y
    def __call__(self,X):
        return self.T(X)

class DistortTransf2D:
    def __init__(self,ri,rf):
        """
        defines a distort matrix from the square defined by the vertices in ri towards the square defined by the vertices in rf

        ri[k] must correspond to rf[k]
        each must have four vertices

        ri is the undistorted square, rf is the distorted square

        ri,rf -> one vertice per row!

        this transform implements a projective transformation matrix between the two frames of reference ri and rf, according to:
        https://math.stackexchange.com/a/339033/59039
        """
        if not self._is_numpy_array(ri):
            ri = numpy.array(ri)
        if not self._is_numpy_array(rf):
            rf = numpy.array(rf)
        self._ri = ri
        self._rf = rf
        self._C = numpy.zeros((3,3))
        self._calc_matrix()
    def _calc_matrix(self):
        # we first solve the linear system
        #     M.u = v, with M,u,v given by
        #   [ xi1   xi2   xi3  ]   [ lambda ]   [ xi4 ]
        #   [ yi1   yi2   yi3  ] . [  tau   ] = [ yi4 ]
        #   [  1     1     1   ]   [   mu   ]   [  1  ]
        # for u=(lambda,tau,mu), to obtain a matrix
        #       [ lambda*xi1   mu*xi2   tau*xi3  ]
        #   A = [ lambda*yi1   mu*yi2   tau*yi3  ]
        #       [ lambda       mu       tau      ]
        # and do the same for xf,yf to obtain a matrix B similarly to A
        get_M = lambda r: numpy.column_stack((r[:3,:],numpy.ones(3,dtype=float))).T
        get_v = lambda r: numpy.append(r[-1,:],1.0)
        A = get_M(self._ri)*numpy.linalg.solve(get_M(self._ri),get_v(self._ri)) # multiplies each column of u with each column of A
        B = get_M(self._rf)*numpy.linalg.solve(get_M(self._rf),get_v(self._rf)) # multiplies each column of u with each column of A
        # now we define the transform matrix C = B.inverse(A)
        self._C = numpy.matmul(B,numpy.linalg.inv(A))
    def T(self,r):
        """
        r must be a single vector or 1 vector per row
        """
        if not self._is_numpy_array(r):
            r = numpy.asarray(r)
        is_1d = r.ndim == 1
        if is_1d:
            r = r[numpy.newaxis,:]
        # now we multiply a position r by the matrix to obtain a transformed position
        r_prime = self._C.dot(numpy.row_stack((r.T,numpy.ones(r.shape[0])))).T
        r_prime = r_prime[:,:2]/r_prime[:,2][:,numpy.newaxis] # and dehomogenize the coordinate
        return r_prime.flatten() if is_1d else r_prime
    def __call__(self,r):
        """
        r must be a single vector or 1 vector per row
        """
        return self.T(r)
    def _is_numpy_array(self,x):
        return type(x) is numpy.ndarray

class LinearTransfFit:
    def __init__(self,X,Y,vectors_in_rows=True,reorder_to_match_min_dist_points=True):
        """
        defines a linear transformation matrix T such that
        TX[i] = Y[i]
        by linear regression

        X               -> set of start vectors
        Y               -> set of obtained vectors
        vectors_in_rows -> if True, each row of X and Y are vectors and number of rows in both X and Y must match; if false, the same applies for columns in X and Y

        reorder_to_match_min_dist_points -> if True, then reorder Y such that Y[i] is closest to X[i], 
        """
        X,Y = self._init_transform(X,Y,vectors_in_rows,reorder_to_match_min_dist_points)
        self._fit(X,Y)
    def _fix_XY(self,X,Y,vectors_in_rows,reorder_to_match_min_dist_points):
        if not self._is_numpy_array(X):
            X = numpy.asarray(X)
        if not self._is_numpy_array(Y):
            Y = numpy.asarray(Y)
        X = X if vectors_in_rows else X.T
        Y = Y if vectors_in_rows else Y.T
        if reorder_to_match_min_dist_points:
            X,Y = self._sort_XY(X,Y)
        return X,Y
    def _sort_XY(self,X,Y):
        added = numpy.zeros(Y.shape[0],dtype=bool)
        Y_new = numpy.zeros(Y.shape)
        N     = X.shape[0]
        for i in range(N):
            d = numpy.linalg.norm(Y - X[i,:],axis=1)
            while True:
                k = numpy.nanargmin( d )
                if added[k]:
                    d[k] = numpy.nan # remove item k because it has been added already
                else:
                    break
            added[k]   = True
            Y_new[i,:] = Y[k,:]
        return X,Y_new
    def _init_transform(self,X,Y,vectors_in_rows,reorder_to_match_min_dist_points):
        X,Y = self._fix_XY(X,Y,vectors_in_rows,reorder_to_match_min_dist_points)
        self._vectors_in_rows = vectors_in_rows
        self._n = X.shape[1] # we transpose in fix_XY and force all vectors to be in rows for the calculations
        self._m = Y.shape[1] # we transpose in fix_XY and force all vectors to be in rows for the calculations
        self._T = numpy.zeros((self._m,self._n))
        return X,Y
    def _fit(self,X,Y):
        self._T = numpy.linalg.pinv(X).dot(Y).T
    def T(self,x):
        if not self._is_numpy_array(x):
            x = numpy.asarray(x)
        if x.ndim == 1:
            return self._T.dot(x)
        result = self._T.dot(x.T if self._vectors_in_rows else x)
        return result.T if self._vectors_in_rows else result
    def __call__(self,x):
        return self.T(x)
    def _is_numpy_array(self,x):
        return type(x) is numpy.ndarray


class LinearTransf2D:
    def __init__(self,X0_lim,X1_lim,Y0_lim,Y1_lim):
        """ Creates a linear transformation from X0 to X1, and Y0 to Y1
        X0_lim -> interval range of X0 (from)
        X1_lim -> interval range of X1 (to)
        Y0_lim -> interval range of Y0 (from)
        Y1_lim -> interval range of Y1 (to)
        """
        self._X0_lim = numpy.asarray(X0_lim).flatten()[0:2]
        self._X1_lim = numpy.asarray(X1_lim).flatten()[0:2]
        self._Y0_lim = numpy.asarray(Y0_lim).flatten()[0:2]
        self._Y1_lim = numpy.asarray(Y1_lim).flatten()[0:2]
        self._Ax = numpy.nan
        self._Bx = numpy.nan
        self._Ay = numpy.nan
        self._By = numpy.nan
        self.CalcCoeff()
    def CalcCoeff(self):
        self._Ax = numpy.diff(self._X1_lim) / numpy.diff(self._X0_lim)
        self._Bx = self._X1_lim[0] - self._Ax * self._X0_lim[0]
        self._Ay = numpy.diff(self._Y1_lim) / numpy.diff(self._Y0_lim)
        self._By = self._Y1_lim[0] - self._Ay * self._Y0_lim[0]
    def T(self,r):
        if not(type(r) is numpy.ndarray):
            r = numpy.asarray(r)
        if (r.ndim > 2) or ((len(r.shape) > 1) and (r.shape[1] > 2)):
            raise ValueError('r can only be a 2d vector, or an N-by-2 array with one 2d vector per row')
        if r.ndim == 1:
            return numpy.asarray((r[0]*self._Ax[0]+self._Bx[0] , r[1]*self._Ay[0]+self._By[0]))
        else:
            return numpy.column_stack((numpy.add(numpy.multiply(r[:,0],self._Ax),self._Bx) , numpy.add(numpy.multiply(r[:,1],self._Ay),self._By)))
    def __call__(self,r):
        return self.T(r)

class ProbabilityType(Enum):
    cumulative_prob = 0
    cumulative_step = 1
    independent = 2
    def __str__(self):
        return str(self._name_)

class structtype(collections.abc.MutableMapping):
    def __init__(self,struct_fields=None,field_values=None,**kwargs):
        if not(type(struct_fields) is type(None)):
            #assert not(type(values) is type(None)),"if you provide field names, you must provide field values"
            if not self._is_iterable(struct_fields):
                struct_fields = [struct_fields]
                field_values = [field_values]
            kwargs.update({f:v for f,v in zip(struct_fields,field_values)})
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
        return self
    def SetAttr(self,field,value):
        if not self._is_iterable(field):
            field = [field]
            value = [value]
        self.__dict__.update({f:v for f,v in zip(field,value)})
        return self
    def GetFields(self):
        return '; '.join([ k for k in self.__dict__.keys() if (k[0:2] != '__') and (k[-2:] != '__') ])
        #return self.__dict__.keys()
    def IsField(self,field):
        return field in self.__dict__.keys()
    def RemoveField(self,field):
        return self.__dict__.pop(field,None)
    def RemoveFields(self,*fields):
        r = []
        for k in fields:
            r.append(self.__dict__.pop(k,None))
        return r
    def KeepFields(self,*fields):
        keys = list(self.__dict__.keys())
        for k in keys:
            if not (k in fields):
                self.__dict__.pop(k,None)
    def __setitem__(self,label,value):
        self.__dict__[label] = value
    def __getitem__(self,label):
        return self.__dict__[label]
    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))
    def _get_kwargs(self):
        return sorted(self.__dict__.items())
    def _get_args(self):
        return []
    def _is_iterable(self,obj):
        return (type(obj) is list) or (type(obj) is tuple)
    def __delitem__(self,*args):
        self.__dict__.__delitem__(*args)
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return iter(self.__dict__)

def is_trackfile(f):
    return type(f) == trackfile

class trackfile(structtype):
    def __init__(self,**kwargs):
        structtype.__init__(self,**kwargs)
    def Set(self,**kwargs):
        if 'arena_picture_extent' in kwargs.keys():
            assert len(kwargs['arena_picture_extent']) == 4, "trackfile :: setting 'arena_picture_extent' must be passed a list with four elements [left, right, bottom, top]"
            arena_pic_left   = kwargs['arena_picture_extent'][0]
            arena_pic_right  = kwargs['arena_picture_extent'][1]
            arena_pic_bottom = kwargs['arena_picture_extent'][2]
            arena_pic_top    = kwargs['arena_picture_extent'][3]
            self = structtype.Set(self,arena_pic_left=arena_pic_left,arena_pic_right=arena_pic_right,arena_pic_bottom=arena_pic_bottom,arena_pic_top=arena_pic_top)
            kwargs.pop('arena_picture_extent',None)
        return structtype.Set(self,**kwargs)
    def arena_picture_extent(self):
        return [self.arena_pic_left, self.arena_pic_right, self.arena_pic_bottom, self.arena_pic_top]
    def __repr__(self):
        type_name          = type(self).__name__
        arg_strings = []
        is_valid_print_arg = lambda name: (name[0] != '_') and (name[:2] != 'r_') and (name[:5] != 'unit_') and (not (name in ['file_name','velocity','time','direction','arena_diameter','file_trial_idx','trial_id','trial_name'])) and (not ('arena_pic' in name))
        arg_strings.append('%s=%r' % ('file_name', os.path.split(self.__dict__['file_name'])[1] ))
        for name, value in self._get_kwargs():
            if is_valid_print_arg(name):
                arg_strings.append('%s=%r' % (name, value))
        arg_strings.append('%s=%r' % ('arena_picture'       , os.path.split(self.__dict__['arena_picture'])[1] ))
        arg_strings.append('%s=%r' % ('arena_picture_extent', [self.__dict__['arena_pic_left'], self.__dict__['arena_pic_right'], self.__dict__['arena_pic_bottom'], self.__dict__['arena_pic_top']] ))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))
    def get_info(self):
        return 'mouse=%s ; trial=%s ; startq=%s'%(self.__dict__['mouse_number'],self.__dict__['trial'],self.__dict__['start_quadrant'])

@jit(nopython=True,fastmath=True,error_model='numpy',cache=True)
def find_self_intersection_jit(r,t):#,interpolate=False,n_points=100):
    """
    finds all self-intersections of the curve r in the plane
    r[t,0] -> x[t]; r[t,1] -> y[t];
    then, for every two pairs of points, an intersection happens
    for a fixed t = t1, such that
    v = r[t1+1,:] - r[t1,:]
    you get an instant t2 where r[t2,:] and r[t2+1,:] are in opposite sides of v
    (i.e., if r[t2+1,:] is to the left of v, then r[t2,:] must be to the right,
    and vice-versa)
    on top of that, we need the intersection between the two vectors r[t2+1,:]-r[t2,:] and r[t1+1,:]-r[t1,:]
    to fall on the line segment between r[t1,:] and r[t1+1,:]

    in order to find the intersection between r[t2+1,:]-r[t2,:] and r[t1+1,:]-r[t1,:],
    we note that these two vectors should form the four vertices of a quadrilateral, and their intersection
    is given by
    https://mathworld.wolfram.com/Line-LineIntersection.html

    The formulas of the wolfram website are given in terms of the three vectors:
                                . r[t1+1,:]
                    .r[t2,:]
                u1 /
                  /
         r[t1,:] .__________________. r[t2+1,:]
                         u2
         u1 = r[t2,:]   - r[t1,:] = (x1-x3,y1-y3) [[ x and y are from the wolfram link ]]
         u2 = r[t2+1,:] - r[t1,:] = (x2-x3,y2-y3)
         v1 = r[t1+1,:] - r[t1,:] = (x4-x3,y4-y3)
    
    the scalar product between u2.v1 and u1.v1 must be positive (such that the intersection is between t1 and t1+1, and not behind it)
    and the norm of the intersection point must be less than the norm of v1

    CAUTION (interpolate not implemented):
    if you choose to interpolate, this algorithm may take VERY long, order of n_points**2*T**2 to complete

    parameters:
        t           -> time vector, must be flatten with size == r.shape[0] (number of rows in r)
        interpolate -> if True, then subdivides each found intersection into n_points to increase the intersection point determination
        n_points    -> number of points to subdivide the intersection interval in order to increase the precision of the intersection calculation

    alternatively, you can use the function
    find_intersection(x1,y1,x2,y2)
    but some trajectories are just too long and that function runs out of memory
    """
    #assert (type(r) is numpy.ndarray) and (r.ndim == 2),"r must be a Tx2 ndarray"
    #if type(t) is type(None):
    #    t = numpy.arange(r.shape[0])
    #if interpolate:
    #    r_func = scipy.interpolate.interp1d(t,r,kind='linear',copy=False,axis=0)
    T = r.shape[0] - 1
    t_inter = []
    r_inter = []
    for t1 in range(T):
        if numpy.any(numpy.isnan(r[t1,:])) or numpy.any(numpy.isnan(r[t1+1,:])):
            continue
        v1 = r[t1+1,:] - r[t1,:]
        if _norm_2d(v1) < 1e-18: # avoiding spots where the trajectory doesnt move
            continue
        for t2 in range(t1+2,T): # +2 because a pair from t to t+1 will never intersect with t+1 to t+2, since they always form a triangle
            if numpy.any(numpy.isnan(r[t2,:])) or numpy.any(numpy.isnan(r[t2+1,:])):
                continue
            u1 = r[t2,:] - r[t1,:]
            u2 = r[t2+1,:] - r[t1,:]
            if _norm_2d(u2 - u1) < 1e-18: # avoiding spots where the trajectory doesnt move
                continue
            p = _intersect_lines(v1,u1,u2)
            if ((_dot_prod_2d(u1,v1)>0.0) and (_dot_prod_2d(u2,v1)>0.0)) and (_norm_2d(p) <= _norm_2d(v1)) and xor(_is_to_the_right_jit(v1,u1),_is_to_the_right_jit(v1,u2)):
                # there is an intersection between t1,t1+1 and t2,t2+1
                # if
                # - both u1 and u2 point in the same direction as v1 (i.e., the at t2 and t2+1 lie beyond the one at t1)
                # - the points at t2 and t2+1 lie at opposite sides of the vector v1
                # - the distance from the point at t1 and the intersection is less than the distance from t1 to t1+1
                #print(t1)
                #if interpolate:
                #    tt = numpy.linspace(t[t1],t[t2+1],n_points)
                #    t_temp,r_temp = find_self_intersection( r_func(tt), t=tt, interpolate=False ) # we dont want to interpolate any deeper to avoid stack overflow
                #    t_inter.extend(t_temp)
                #    r_inter.extend(r_temp)
                #else:
                #    t_inter.append((t[t2] + t[t2+1])/2.0) # the intersection happens in the future, not in t1
                #    # the intersection happens at the centroid of the four coordinates
                #    r_inter.append( nanmean(r[[t1,t1+1,t2,t2+1],:],axis=0) )
                t_inter.append((t[t2] + t[t2+1])/2.0) # the intersection happens in the future, not in t1
                r_inter.append( (r[t1,0]+p[0], r[t1,1]+p[1]) )
    if len(t_inter) > 1:
        ind_sort = numpy.argsort(numpy.array(t_inter,dtype=numpy.float64))
        t_inter = [ t_inter[k] for k in ind_sort ]
        r_inter = [ r_inter[k] for k in ind_sort ]
        return t_inter,r_inter
    else:
        return t_inter,r_inter

@jit(nopython=True,fastmath=True,error_model='numpy',cache=True)
def xor(A,B):
    return (A and (not B)) or ((not A) and B)

@jit(nopython=True,fastmath=True,error_model='numpy',cache=True)
def _intersect_lines(v1,u1,u2):
    """
    calculates intersection between the lines defined by the vectors
    u2 - u1 and v1,
    https://mathworld.wolfram.com/Line-LineIntersection.html

    The formulas of the wolfram website are given in terms of the three vectors:
                                . r[t1+1,:]
                    .r[t2,:]
                u1 /
                  /
         r[t1,:] .__________________. r[t2+1,:]
                         u2
         u1 = r[t2,:]   - r[t1,:] = (x1-x3,y1-y3) [[ x and y are from the wolfram link ]]
         u2 = r[t2+1,:] - r[t1,:] = (x2-x3,y2-y3)
         v1 = r[t1+1,:] - r[t1,:] = (x4-x3,y4-y3)

         anchor all vectors to the origin, intersection point is returned relative to the origin
    
    returns
        intersection point coordinate (tuple) relative to r[t1,:] (i.e., to the 0 of v1)
    """
    p = u1[1]*u2[0]-u1[0]*u2[1]
    d = (u2[0]-u1[0])*v1[1]-(u2[1]-u1[1])*v1[0]
    return numpy.array([p*v1[0]/d,p*v1[1]/d])


@jit(nopython=True,fastmath=True,error_model='numpy',cache=True)
def _is_to_the_right_jit(u,v):
    """
    returns True if v is to the right of u
    (u and v are two vectors on the x,y plane; list, tuple, numpy.ndarray with ndim==1 and size==2)
    """
    return _dot_prod_2d(u,(-v[1],v[0])) > 0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v

@jit(nopython=True,fastmath=True,error_model='numpy',cache=True)
def _norm_2d(u):
    return math.sqrt(u[0]*u[0] + u[1]*u[1])

@jit(nopython=True,fastmath=True,error_model='numpy',cache=True)
def _dot_prod_2d(u,v):
    return u[0]*v[0] + u[1]*v[1]

#@jit(nopython=True,parallel=True, fastmath=True)
def find_neighbor_points(r,precision=1e-10):
    """
    finds all the neighboring points of each point in r within precision
    meaning, for each point (x,y) in r,
    find all the other points in r that are close to (x,y)
    within precision
    and are not the former and next points in t-1 and t+1

    r == T x 2 ndarray, T rows -> defines time,
    such that x[t],y[t] = r[t,:]

    returns
        * k_inter -> indices of all self-intersections
        * r[k_inter,:] -> positions of all self-intersections
    """
    assert (type(r) is numpy.ndarray) and (r.ndim == 2),"r must be a Tx2 ndarray"
    k_inter = []
    for t in range(r.shape[0]):
        r0 = r[t,:]
        # we don't check itself, neither the next,
        # nor the previous, such that we determine only unique intersections
        p = find_contiguous_pieces(numpy.all(numpy.abs(r[(t+1):,:] - r0) < precision,axis=1))
        k = [ int(numpy.median(kk)) for kk in p ] # indices of all the intersections with r0
        k_inter.extend(k)
    return k_inter,r[k_inter,:]

#@jit(nopython=True,parallel=True, fastmath=True)
def find_contiguous_pieces(cond):
    """
    finds all the sequential cond==True that are separated by at least one cond==False
    
    returns a list where each entry contains
            the indices where cond has sequential True's
    """
    if len(cond) == 0:
        return []
    f = numpy.logical_xor(cond[:-1],cond[1:])
    k_start = numpy.nonzero(numpy.logical_and( f , cond[1:]  ))[0] + 1 # index of the start of a piece
    k_end   = numpy.nonzero(numpy.logical_and( f , cond[:-1] ))[0] + 1 # index of the end of a piece
    if cond[0]:
        k_start = numpy.insert(k_start,0,0)
    if cond[-1]:
        k_end = numpy.insert(k_end,k_end.size,len(cond))
    if (k_start.size > 0) and (k_end.size > 0):
        return [ numpy.arange(a,b) for a,b in zip(k_start,k_end) ]
    return []

#@jit(nopython=True,parallel=True, fastmath=True)
def find_intersection(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = numpy.linspace(3, 10, 100)
    x1 = a*phi - b*numpy.sin(phi)
    y1 = a - b*numpy.cos(phi)
    x2=phi    
    y2=numpy.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    def _rect_inter_inner(x1,x2):
        n1=x1.shape[0]-1
        n2=x2.shape[0]-1
        X1=numpy.c_[x1[:-1],x1[1:]]
        X2=numpy.c_[x2[:-1],x2[1:]]    
        S1=numpy.tile(X1.min(axis=1),(n2,1)).T
        S2=numpy.tile(X2.max(axis=1),(n1,1))
        S3=numpy.tile(X1.max(axis=1),(n2,1)).T
        S4=numpy.tile(X2.min(axis=1),(n1,1))
        return S1,S2,S3,S4

    def _rectangle_intersection_(x1,y1,x2,y2):
        S1,S2,S3,S4=_rect_inter_inner(x1,x2)
        S5,S6,S7,S8=_rect_inter_inner(y1,y2)

        C1=numpy.less_equal(S1,S2)
        C2=numpy.greater_equal(S3,S4)
        C3=numpy.less_equal(S5,S6)
        C4=numpy.greater_equal(S7,S8)

        ii,jj=numpy.nonzero(C1 & C2 & C3 & C4)
        return ii,jj

    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=numpy.diff(numpy.c_[x1,y1],axis=0)
    dxy2=numpy.diff(numpy.c_[x2,y2],axis=0)

    T=numpy.zeros((4,n))
    AA=numpy.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=numpy.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=numpy.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=numpy.nan


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]

#def is_function(x):
#    return type(x) is type(lambda:1)

#def is_scipy_spline(x):
#    return type(x) is scipy.interpolate.interp1d

#def is_func_or_spline(x):
#    return is_function(x) or is_scipy_spline(x)

def find_inter_func(x,y1,y2,useSpline=True,useSimpleMethod=False):
    if useSimpleMethod:
        k = numpy.nonzero((y1[1:] - y2[1:])*(y1[:-1] - y2[:-1]) <= 0)[0]+1
        k[k<0] = 0
        k[k>=len(x)] = len(x)
        x0 = (x[k] + x[k-1])/2.0
        y0 = (y1[k]+y2[k]+y1[k-1]+y2[k-1])/4.0
        return x0,y0
    if callable(y1) and callable(y2):
        try:
            x0,_ = find_intersection(x,y1(x),x,y2(x))
        except:
            return find_inter_func(x,y1(x),y2(x),useSpline=False,useSimpleMethod=True)
        x0 = scipy.optimize.newton( lambda xx: y1(xx) - y2(xx), x0 )
        return x0,(y1(x0)+y2(x0))/2.0
    else:
        if numpy.isscalar(y1):
            yy1 = y1*numpy.ones(x.shape)
        else:
            yy1 = y1
        if numpy.isscalar(y2):
            yy2 = y2*numpy.ones(x.shape)
        else:
            yy2 = y2
        if useSpline:
            if not callable(yy1):
                yy1 = scipy.interpolate.interp1d(x,yy1,kind='cubic',copy=False)
            if not callable(yy2):
                yy2 = scipy.interpolate.interp1d(x,yy2,kind='cubic',copy=False)
            return find_inter_func(x,yy1,yy2)
        else:
            if callable(yy1):
                yy1 = yy1(x)
            if callable(yy2):
                yy2 = yy2(x)
            try:
                return find_intersection(x,yy1,x,yy2)
            except:
                return find_inter_func(x,yy1,yy2,useSpline=False,useSimpleMethod=True)

#@jit(nopython=True,parallel=True, fastmath=True)
def contains_nan(x):
    if _is_numpy_array(x):
        if x.size == 0:
            return False
    else:
        if len(x) == 0:
            return False
    k = numpy.argmax(numpy.isnan(x).astype(numpy.int64))
    return not( (k == 0) and (not numpy.isnan(x[0])))

#@jit(nopython=True,parallel=True, fastmath=True)
def find_last(condition):
    return find_first(condition[::-1])

#@jit(nopython=True,parallel=True, fastmath=True)
def find_first(condition):
    if condition.size == 0:
        return -1
    k = numpy.argmax(condition.astype(numpy.int64))
    if (k == 0) and (not condition[0]):
        return -1 # no element found
    else:
        return k

#@jit(nopython=True,parallel=True, fastmath=True)
def avg_count_nan(x,axis=0):
    """
    averages x over axis
    x is a 2d numpy.ndarray

    ex:
    x = [
        [1,2,3,nan,nan],
        [4,5,6,7,nan],
        [8,9,10,11,12]
        ]
    
    avg_count_nan(x,axis=0)
    averages x over rows (i.e., for each col):
    x_avg = nanmean(x,axis=axis)
    n_valid = [3,3,3,2,1] # number of valid entries for each average
    n_valid = numpy.sum(numpy.logical_not(numpy.isnan(x)).astype(int),axis=axis)
    
    returns
        - avg of x (ignoring nan's)
        - std of x (ignoring nan's)
        - number of valid entries (not nan) for each avg over axis
    """
    s = nanstd(x,axis=axis)
    n = numpy.sum(numpy.logical_not(numpy.isnan(x)).astype(numpy.int64),axis=axis)
    return nanmean(x,axis=axis), s, s/numpy.sqrt(n), n

def zscore_to_control(X,X_control,axis=0):
    """
    X and X_control are observations of two different groups,
    this function calculates the Z-score of the control group over axis (via zscore function)
    and then calculates the Z-score of X relative to the mean and stddev of X_control

    X and X_controls are matrices of observations vs. features.
    observations must be at the given axis, and features on the other axis
    the number of features must match
    
    e.g., if axis == 0, then observations of X and X_control are given in their rows
                        thus, the number of columns in both X and X_control must match

    to get p-values:
        import scipy.stats as sst
        normal_distribution = sst.norm(loc=0,scale=1.) #loc is the mean, scale is the variance.
        # The normal CDF
        p_values = normal_distribution.cdf(z_values)

    parameters:
        X         -> matrix of observations
        X_control -> matrix of observations for the control group
        axis      -> axis of X and X_control observations
    
    returns:
        Z         -> Z-score of X relative to X_control
        Z_control -> Z-score of X_control
    """
    assert (type(X) is numpy.ndarray) and (type(X_control) is numpy.ndarray), "X and X_control must be numpy.ndarray"
    assert (axis == 0) or (axis == 1), "axis must be 0 (rows) or 1 (cols) for observations"
    assert (X.ndim <= 2) and (X_control.ndim <= 2), "X and X_control can only be 2-d matrices of features and observations, or 1-d vectors of observations"
    if X.ndim != X_control.ndim:
        if (X.ndim == 1) and (X_control.ndim == 2): # observations of a single feature for X will be compared to each control feature
            if axis == 0:
                X = numpy.tile(X.reshape((X.size,1)),(1,X_control.shape[1]))
            else:
                X = numpy.tile(X.flatten(),(X_control.shape[0],1))
        if (X_control.ndim == 1) and (X.ndim == 2): # each feature of X will be compared to a single control feature
            if axis == 0:
                X_control = numpy.tile(X_control.reshape((X_control.size,1)),(1,X.shape[1]))
            else:
                X_control = numpy.tile(X_control.flatten(),(X.shape[0],1))
    if (X.ndim > 1) and (X_control.ndim > 1):
        axis_features = 1-axis
        assert X.shape[axis_features] == X_control.shape[axis_features], "both X and X_control must have the same number of features, or one of them must be a single feature"
    
    # both are flat vectors of observations, then Z and Z-score are scalars
    if (X.ndim == 1) and (X_control.ndim == 1):
        Z_control,m_control,s_control = zscore(X_control,return_mean_std=True)
        Z = (X - m_control) / s_control
        return Z,Z_control 

    # otherwise
    if axis == 1: # forcing observations to be rows
        X         = X.T
        X_control = X_control.T
    
    # calculating the z-scores
    Z_control,m_control,s_control = zscore(X_control,axis=0,return_mean_std=True)
    if X_control.shape[0] != X.shape[0]: # the number of observations in X and X_control don't match
        m_control = _match_to_shape(m_control,X,axis=0)
        s_control = _match_to_shape(s_control,X,axis=0)
    Z = (X - m_control) / s_control
    # returning
    if axis == 1: # returning to the desired shape
        Z         = Z.T
        Z_control = Z_control.T
    return Z,Z_control

def _match_to_shape(A,B,axis=0):
    """
    returns A such that A.shape[axis] == B.shape[axis]
    repeats A over axis if need B, or slice A to B size if A is larger
    """
    assert (type(A) is numpy.ndarray) and (type(B) is numpy.ndarray), "A and B must be numpy.ndarray"
    assert (axis == 0) or (axis==1), "axis must be 0 or 1"
    assert (A.ndim == 2) and (B.ndim == 2), "A and B must be matrices"
    if axis == 1: # forcing to match rows
        A = A.T
        B = B.T
    nA = A.shape[0]
    nB = B.shape[0]
    if nB > nA:
        A = numpy.tile(A,(int(numpy.ceil(float(nA)/float(nB))),1))
    A = A[:nB,:]
    if axis == 1: # forcing to match rows
        A = A.T
    return A

def zscore(X,axis=0,return_mean_std=False):
    """
    calculates the z-score of X matrix over the given axis

    X is a matrix of features vs. observations,
    such that the observations lie on the given axis, and features on the other axis
    e.g., if axis == 0, each row of X is a different observation (i.e., means and stddev are taken over rows)
                        and each column of X is a different feature

    to get p-values:
        import scipy.stats as sst
        normal_distribution = sst.norm(loc=0,scale=1.) #loc is the mean, scale is the variance.
        # The normal CDF
        p_values = normal_distribution.cdf(z_values)

    parameters: 
        X    -> matrix of observations
        axis -> axis of X observations;
    returns Z(X) = (X-mean(X))/std(X)
    """
    if not(type(X) is numpy.ndarray):
        X = numpy.asarray(X)
    if X.ndim > 1:
        m = nanmean(X,axis=axis)
        s = nanstd(X,axis=axis)
    else:
        m = nanmean(X)
        s = nanstd(X)
    s[s==0.0] = 1.0
    if X.ndim > 1:
        if axis == 0:
            # repeats the row-average over the rows of x
            m = numpy.tile(m.flatten(),(X.shape[0],1))
            s = numpy.tile(s.flatten(),(X.shape[0],1))
        elif axis == 1:
            # repeats the col-average over the cols of x
            m = numpy.tile(m.reshape((m.size,1)),(1,X.shape[1]))
            s = numpy.tile(s.reshape((s.size,1)),(1,X.shape[1]))
        else:
            raise ValueError('zscore not implemented for axis > 1')
    if return_mean_std:
        return (X - m) / s,m,s
    else:
        return (X - m) / s

def _is_scipy_linreg_or_ttest_result(x):
    return ('Ttest_indResult' in str(type(x))) or ('LinregressResult' in str(type(x)))

def ttest(X_Controls, X_Test, zscore_to_controls=False,axis=0,**ttest_ind_args):
    """
    wrapper for scipy.stats.ttest_ind function

    X_Controls         -> control group sample
    X_Test             -> test group sample
    axis               -> if 0 (default; observations in rows, variables in columns);
                          if 1 (observations in columns, variables in rows)
    zscore_to_controls -> if True, compares the Z-Scores of X_Test and X_Controls,
                          both calculated from X_Controls mean and stddev
    """
    if zscore_to_control:
        X_Test,X_Controls = zscore_to_control(X_Test,X_Controls,axis=axis)
    return scipy.stats.ttest_ind(X_Controls,X_Test,axis=axis,**ttest_ind_args)

def check_p_values(p_or_ttest_result,p_threshold,q_threshold=None):
    """
    checks whether p < p_threshold
    if q_threshold is set, then calculates FDR for the p values
    and checks whether p < p_FDR
    """
    if type(p_or_ttest_result) is list:
        if (len(p_or_ttest_result) > 0):
            if all(_is_scipy_linreg_or_ttest_result(pp) for pp in p_or_ttest_result):
                p_or_ttest_result = numpy.array([ pp.pvalue for pp in  p_or_ttest_result ])
                #print(p_or_ttest_result)
        else:
            return False
    if _is_scipy_linreg_or_ttest_result(p_or_ttest_result):
        p = p_or_ttest_result.pvalue
    else:
        p = p_or_ttest_result
    p = p.data if type(p) is numpy.ma.core.MaskedArray else p
    if not(type(p) is numpy.ndarray):
        p = numpy.asarray(p)
    if (p.size == 1):
        p = p.flatten()[0]
    if numpy.isscalar(p):
        return p < p_threshold
    if not(type(q_threshold) is type(None)):
        p_FDR       = get_FDR_p_threshold(p,q_threshold,'indep')
        p_threshold = p_FDR if p_FDR < p_threshold else p_threshold
    return p < p_threshold

def get_FDR_p_threshold(p,q,p_type='indep'):
    """
    p_type == 'indep' or 'nonpar'

    % FORMAT [pID,pN] = FDR(p,q)
    % 
    % p   - vector of p-values
    % q   - False Discovery Rate level
    %
    % pID - p-value threshold based on independence or positive dependence
    % pN  - Nonparametric p-value threshold
    %______________________________________________________________________________
    % $Id: FDR.m,v 2.1 2010/08/05 14:34:19 nichols Exp $
    """
    p_type = p_type.lower()
    assert p_type in ['indep','nonpar'],"p_type must be indep or nonpar"
    p = p[numpy.logical_not(numpy.isnan(p))].flatten() # Toss NaN's
    p = numpy.sort(p)
    V = p.size
    I = numpy.arange(1,V+1)
    cVID = 1
    cVN  = numpy.sum(1/I)
    #pID = p(  find(p<=I./V.*q./cVID, 1, 'last' )  );
    k = find_last( p <= ((I/V)*(q/cVID))  )
    pID = p[k] if k >= 0 else 0.0
    #pN = p(find(p<=I./V.*q./cVN, 1, 'last' ));
    k = find_last( p <= (I/V)*(q/cVN) )
    pN = p[k] if k >= 0 else 0.0
    if p_type == 'indep':
        return pID
    else:
        return pN

def get_finite(x):
    return x[numpy.isfinite(x)]

def nanmean(x,**nanmeanargs):
    if numpy.isscalar(x):
        x = numpy.array([x])
    x_masked = numpy.ma.masked_invalid(x)
    if x_masked.count() == 0:
        if x.size == 0:
            return numpy.array([])
        x_m = numpy.nanmean(x,**nanmeanargs)
        if numpy.isscalar(x_m):
            return numpy.nan
        shape = x_m.shape
        if 'axis' in nanmeanargs.keys():
            shape[nanmeanargs['axis']] = 0
        else:
            shape = (0,)
        return numpy.empty(shape=shape)
    s = numpy.nanmean(x_masked,**nanmeanargs)
    return s.data if type(s) is numpy.ma.core.MaskedArray else s

def nanstd(x,**nanstdargs):
    if numpy.isscalar(x):
        x = numpy.array([x])
    x_masked = numpy.ma.masked_invalid(x)
    if x_masked.count() == 0:
        if x.size == 0:
            return numpy.array([])
        x_m = numpy.nanstd(x,**nanstdargs)
        if numpy.isscalar(x_m):
            return numpy.nan
        shape = x_m.shape
        if 'axis' in nanstdargs.keys():
            shape[nanstdargs['axis']] = 0
        else:
            shape = (0,)
        return numpy.empty(shape=shape)
    s = numpy.nanstd(x_masked,**nanstdargs)
    return s.data if type(s) is numpy.ma.core.MaskedArray else s

def nanserr(x,**nanstdargs):
    if numpy.isscalar(x):
        x = numpy.array([x])
    #if not _is_numpy_array(x):
    #    x = numpy.asarray(x)
    nanstdargs = set_default_kwargs(nanstdargs,ddof=1)
    axis = None
    if 'axis' in nanstdargs.keys():
        axis = nanstdargs['axis']
    x_masked = numpy.ma.masked_invalid(x)
    ndim = x_masked.ndim
    if x_masked.count() == 0:
        if x.size == 0:
            return numpy.array([])
        x_m = numpy.nanstd(x,**nanstdargs)
        if numpy.isscalar(x_m):
            return numpy.nan
        shape = x_m.shape
        if 'axis' in nanstdargs.keys():
            shape[nanstdargs['axis']] = 0
        else:
            shape = (0,)
        return numpy.empty(shape=shape)
    s = numpy.nanstd(x_masked,**nanstdargs)
    if ndim == 1:
        n = numpy.sqrt(numpy.sum((~x_masked.mask).astype(float)))
    else:
        n = numpy.sqrt(numpy.sum((~x_masked.mask).astype(float),axis=axis))
    return (s.data if type(s) is numpy.ma.core.MaskedArray else s)/n

def nanmin(x,**nanminargs):
    if numpy.isscalar(x):
        x = numpy.array([x])
    x_masked=numpy.ma.masked_invalid(x)
    if x_masked.count() == 0:
        if x.size == 0:
            return numpy.array([])
        x_m = numpy.nanmin(x,**nanminargs)
        if numpy.isscalar(x_m):
            return numpy.nan
        shape = x_m.shape
        if 'axis' in nanminargs.keys():
            shape[nanminargs['axis']] = 0
        else:
            shape = (0,)
        return numpy.empty(shape=shape)
    s = numpy.nanmin(x_masked,**nanminargs)
    return s.data if type(s) is numpy.ma.core.MaskedArray else s

def nanmax(x,**nanmaxargs):
    if numpy.isscalar(x):
        x = numpy.array([x])
    x_masked=numpy.ma.masked_invalid(x)
    if x_masked.count() == 0:
        if x.size == 0:
            return numpy.array([])
        x_m = numpy.nanmax(x,**nanmaxargs)
        if numpy.isscalar(x_m):
            return numpy.nan
        shape = x_m.shape
        if 'axis' in nanmaxargs.keys():
            shape[nanmaxargs['axis']] = 0
        else:
            shape = (0,)
        return numpy.empty(shape=shape)
    s = numpy.nanmax(x_masked,**nanmaxargs)
    return s.data if type(s) is numpy.ma.core.MaskedArray else s

def linregress_col(x,y,**linregress_args):
    """
    a wrapper of linregress below, where we test each column of x versus the corresponding column in y
    """
    x = x if _is_numpy_array(x) else numpy.asarray(x)
    y = y if _is_numpy_array(y) else numpy.asarray(y)

    if x.ndim == 1:
        if y.ndim != 1:
            raise ValueError('y must be a 1d numpy.ndarray because x is a 1d numpy.ndarray')
        return linregress(x,y,**linregress_args)
    else:
        if x.shape[1] != y.shape[1]:
            raise ValueError('x and y must have the same number of columns')
        return_linear_func = linregress_args['return_linear_func'] if 'return_linear_func' in linregress_args.keys() else False
        r_lst = [ linregress(x[:,k], y[:,k], **linregress_args) for k in range(x.shape[1]) ]
        if return_linear_func:
            return unpack_list_of_tuples(r_lst)
        else:
            return r_lst

def linregress(x,y=None,alternative='two-sided',return_linear_func=False):
    """
    a wrapper for scipy.stats.linregress
    where we get rid of inf and nan from x and y
    """
    has_y = not(type(y) is type(None))
    get_y = lambda yy,k: yy[k] if has_y else None
    is_valid = numpy.isfinite(x)
    if has_y:
        is_valid = numpy.logical_and(is_valid,numpy.isfinite(y))
    null_linregress = structtype(slope            = numpy.nan,
                                 intercept        = numpy.nan,
                                 rvalue           = numpy.nan,
                                 pvalue           = numpy.nan,
                                 stderr           = numpy.nan,
                                 intercept_stderr = numpy.nan)
    s = null_linregress
    try:
        s = scipy.stats.linregress(x[is_valid],get_y(y,is_valid),alternative=alternative)
    except:
        pass
    if return_linear_func:
        f = lambda x,linreg: linreg.intercept + linreg.slope * x
        s = (s,f)
    return s

def get_items_by_index(lst,ind):
    #from operator import itemgetter
    items = operator.itemgetter( *ind  )(lst)
    if type(items) is tuple:
        items = list(items)
    if not(type(items) is list):
        items = [items]
    return items

def jackknife_track_sample(input_tracks):
    """
    input_tracks is a list of track files containing N mice in each trial
    (if it is a list of lists -- meaning it is grouped by mouse of trial), the input list will be flattened
    this function returns N track file lists, each of which has a different mouse removed compared to the input_track list (leave one out procedure)

    each trial must have the same number of mice, otherwise the code will fail
    """
    if is_list_of_1d_collection_or_none(input_tracks):
        input_tracks = list(flatten_list(input_tracks, only_lists=True))
    input_tracks = io.group_track_list(input_tracks,group_by='trial')[0]
    N = numpy.max( [ len(tr_group) for tr_group in input_tracks ] ) # max number of mice
    
    #assert numpy.all( numpy.array([ len(tr_group) for tr_group in input_tracks ]) == N ), 'All trials must have the same number of mice'

    input_tracks_jk_group = []
    for k in range(N):
        g = copy.deepcopy(input_tracks)
        for n,trial_group in enumerate(g):
            ind  = list(  set(range(len(trial_group))) - set([k%len(trial_group)])  )
            g[n] = get_items_by_index(trial_group,ind) # removing mouse k out of N
        input_tracks_jk_group.append(list(flatten_list(g,only_lists=True)))

    return input_tracks_jk_group

"""
The following functions: jackknife_resampling and jackknife_stats, have been copied from the astropy project.
Thus, the 3-clause BSD License follows below

Copyright (c) 2011-2022, Astropy Developers

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    - Neither the name of the Astropy Team nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
def jackknife_resampling(data):
    """Performs jackknife resampling on numpy arrays.

    Jackknife resampling is a technique to generate 'n' deterministic samples
    of size 'n-1' from a measured sample of size 'n'. Basically, the i-th
    sample, (1<=i<=n), is generated by means of removing the i-th measurement
    of the original sample. Like the bootstrap resampling, this statistical
    technique finds applications in estimating variance, bias, and confidence
    intervals.

    Parameters
    ----------
    data : ndarray
        Original sample (1-D array) from which the jackknife resamples will be
        generated.

    Returns
    -------
    resamples : ndarray
        The i-th row is the i-th jackknife sample, i.e., the original sample
        with the i-th measurement deleted.

    References
    ----------
    .. [1] McIntosh, Avery. "The Jackknife Estimation Method".
        <https://arxiv.org/abs/1606.00497>

    .. [2] Efron, Bradley. "The Jackknife, the Bootstrap, and other
        Resampling Plans". Technical Report No. 63, Division of Biostatistics,
        Stanford University, December, 1980.

    .. [3] Jackknife resampling <https://en.wikipedia.org/wiki/Jackknife_resampling>
    """
    n = data.shape[0]
    if n <= 0:
        raise ValueError("data must contain at least one measurement.")

    resamples = numpy.empty([n, n - 1])

    for i in range(n):
        resamples[i] = numpy.delete(data, i)

    return resamples

def jackknife_stats(data, statistic, confidence_level=0.95):
    """Performs jackknife estimation on the basis of jackknife resamples.

    This function requires `SciPy <https://www.scipy.org/>`_ to be installed.

    Parameters
    ----------
    data : ndarray
        Original sample (1-D array).
    statistic : function
        Any function (or vector of functions) on the basis of the measured
        data, e.g, sample mean, sample variance, etc. The jackknife estimate of
        this statistic will be returned.
    confidence_level : float, optional
        Confidence level for the confidence interval of the Jackknife estimate.
        Must be a real-valued number in (0,1). Default value is 0.95.

    Returns
    -------
    estimate : float or `~numpy.ndarray`
        The i-th element is the bias-corrected "jackknifed" estimate.

    bias : float or `~numpy.ndarray`
        The i-th element is the jackknife bias.

    std_err : float or `~numpy.ndarray`
        The i-th element is the jackknife standard error.

    conf_interval : ndarray
        If ``statistic`` is single-valued, the first and second elements are
        the lower and upper bounds, respectively. If ``statistic`` is
        vector-valued, each column corresponds to the confidence interval for
        each component of ``statistic``. The first and second rows contain the
        lower and upper bounds, respectively.

    Examples
    --------
    1. Obtain Jackknife resamples:

    >>> import numpy as np
    >>> from astropy.stats import jackknife_resampling
    >>> from astropy.stats import jackknife_stats
    >>> data = np.array([1,2,3,4,5,6,7,8,9,0])
    >>> resamples = jackknife_resampling(data)
    >>> resamples
    array([[2., 3., 4., 5., 6., 7., 8., 9., 0.],
           [1., 3., 4., 5., 6., 7., 8., 9., 0.],
           [1., 2., 4., 5., 6., 7., 8., 9., 0.],
           [1., 2., 3., 5., 6., 7., 8., 9., 0.],
           [1., 2., 3., 4., 6., 7., 8., 9., 0.],
           [1., 2., 3., 4., 5., 7., 8., 9., 0.],
           [1., 2., 3., 4., 5., 6., 8., 9., 0.],
           [1., 2., 3., 4., 5., 6., 7., 9., 0.],
           [1., 2., 3., 4., 5., 6., 7., 8., 0.],
           [1., 2., 3., 4., 5., 6., 7., 8., 9.]])
    >>> resamples.shape
    (10, 9)

    2. Obtain Jackknife estimate for the mean, its bias, its standard error,
    and its 95% confidence interval:

    >>> test_statistic = np.mean
    >>> estimate, bias, stderr, conf_interval = jackknife_stats(
    ...     data, test_statistic, 0.95)
    >>> estimate
    4.5
    >>> bias
    0.0
    >>> stderr  # doctest: +FLOAT_CMP
    0.95742710775633832
    >>> conf_interval
    array([2.62347735,  6.37652265])

    3. Example for two estimates

    >>> test_statistic = lambda x: (np.mean(x), np.var(x))
    >>> estimate, bias, stderr, conf_interval = jackknife_stats(
    ...     data, test_statistic, 0.95)
    >>> estimate
    array([4.5       ,  9.16666667])
    >>> bias
    array([ 0.        , -0.91666667])
    >>> stderr
    array([0.95742711,  2.69124476])
    >>> conf_interval
    array([[ 2.62347735,   3.89192387],
           [ 6.37652265,  14.44140947]])

    IMPORTANT: Note that confidence intervals are given as columns
    """
    # jackknife confidence interval
    if not (0 < confidence_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    # make sure original data is proper
    n = data.shape[0]
    if n <= 0:
        raise ValueError("data must contain at least one measurement.")

    # Only import scipy if inputs are valid
    from scipy.special import erfinv

    resamples = jackknife_resampling(data)

    stat_data = statistic(data)
    jack_stat = numpy.apply_along_axis(statistic, 1, resamples)
    mean_jack_stat = numpy.mean(jack_stat, axis=0)

    # jackknife bias
    bias = (n - 1) * (mean_jack_stat - stat_data)

    # jackknife standard error
    std_err = numpy.sqrt((n - 1) * numpy.mean((jack_stat - mean_jack_stat) * (jack_stat - mean_jack_stat), axis=0))

    # bias-corrected "jackknifed estimate"
    estimate = stat_data - bias

    z_score  = numpy.sqrt(2.0) * erfinv(confidence_level)
    conf_interval = estimate + z_score * numpy.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


def jackknife_func(x, func, return_jk_confint_se=False, return_bias=False, confidence_level=0.95):
    """
    wrapper for jackknife_stats above
    """
    if not(type(x) is numpy.ndarray):
        x         = numpy.asarray(x)
    if x.ndim > 1:
        warnings.warn('jackknife_func :: x :: flattening input data')
        x = x.flatten()
    m, bias, std_err, conf_interval = jackknife_stats(x,func,confidence_level=confidence_level)
    result = m
    if return_jk_confint_se:
        result = (m,conf_interval,std_err)
    if return_bias:
        result += (bias,)
    return result


def bootstrap_func(x,func,return_bs_confint_se=False,**bootstrapArgs):
    """
    this function repeatedly (n_resamples times) applies func(x) along the specified axis (if any)
    and then returns the mean value of func(x) over these n_resamples samples

    delegate parameters to scipy.stats.bootstrap:
        axis             -> axis along which to apply func
        n_resamples      -> number of times we draw a sample from x along axis
        vectorized       -> parallel computations
        confidence_level -> confidence interval within which we calculate func(x)

    example:
    >>> x = 10*numpy.random.randn(1000) # stddev of x == 10
    >>> bootstrap_func(x,numpy.std) # returns approx 10
    """
    if not(type(x) is numpy.ndarray):
        x         = numpy.asarray(x)
    bootstrapArgs = set_default_kwargs(bootstrapArgs,n_resamples=10,vectorized=True,confidence_level=0.95)
    is_1d         = False
    if x.ndim == 1:
        is_1d                 = True
        x                     = x.reshape((x.size,1))
        bootstrapArgs['axis'] = 0
    bs       = scipy.stats.bootstrap((x,),func,**bootstrapArgs)
    get_elem = lambda el: el[0] if is_1d else el
    m        = (bs.confidence_interval.low + bs.confidence_interval.high)/2.0
    m        = get_elem(m)
    confint  = (get_elem(bs.confidence_interval.low), get_elem(bs.confidence_interval.high))
    s        = get_elem(bs.standard_error)
    result   = m
    if return_bs_confint_se:
        result = (m,confint,s)
    return result

#@jit(nopython=True,parallel=True, fastmath=True)
def asarray_nanfill(x):
    """ x is a list of lists or ndarray's
        each element in x is converted to a row in the resulting ndarray
        each row in the resulting ndarray is completed with nan entries in order to match the number of elements
        in the x list that has max number of elements """
    return asarray_fill(x,v=numpy.nan)

#@jit(nopython=True,parallel=True, fastmath=True)
def asarray_fill(x,v=None):
    """ x is a list of lists or ndarray's
        each element in x is converted to a row in the resulting ndarray
        each row in the resulting ndarray is completed with nan entries in order to match the number of elements
        in the x list that has max number of elements """
    v = numpy.nan if v is None else v
    if type(x) is numpy.ndarray:
        return x
    if type(x) is list:
        if (type(x[0]) is int) or (type(x[0]) is float) or numpy.isscalar(x[0]):
            return numpy.asarray(x)
    if not (type(x) is list):
        raise ValueError('x must be a list or an ndarray')
    N = numpy.max([len(xx) for xx in x])
    return numpy.asarray([ fill_to_complete(xx,numpy.nan,N) for xx in x ])

def fill_to_complete(x,v,N):
    """fills x with values v until len(x) == N"""
    if numpy.isscalar(x) or (x is None):
        x = copy.copy(x)
        x = repeat_to_complete([x],N)
    if type(x) is list:
        if N == len(x):
            return x
        y = get_empty_list(N,v)
        y[:len(x)] = x.copy()
        return y
    else:
        if N == len(x):
            return numpy.asarray(x).flatten()
        return numpy.concatenate((numpy.asarray(x).flatten(),numpy.full(N-len(x),v))).flatten()

def get_empty_list(n,v=None):
    return [v for i in range(n)]

def repeat_to_complete(x,N,copydata=False):
# y = [ x(:)', x(:)', ..., x(:)' ]; such that y is [1,N] vector
    getX = lambda r: r
    if copydata:
        getX = lambda r: copy.deepcopy(r)
    if not(type(x) is list) and not(type(x) is numpy.ndarray):
        x = [getX(x)]
    if len(x) == 0:
        return x
    n = len(x)
    m = int(numpy.floor(N/n))
    if m < 1:
        return getX(x[:N])
    if type(x) is list:
        y = get_empty_list(m*n+N%n)
    elif type(x) is numpy.ndarray:
        y = numpy.zeros(m*n+N%n)
    for i in range(m):
        y[i*n:(i*n+n)] = getX(x)
    y[(m*n):] = getX(x[:(N-m*n)])
    return y

def set_default_kwargs(kwargs_dict,**default_args):
    """
    kwargs_dict is the '**kwargs' argument of any function
    this function checks whether any argument in kwargs_dict has a default value given in default_args...
    if yes, and the corresponding default_args key is not in kwargs_dict, then includes it there

    this is useful to avoid duplicate key errors
    """
    kwargs_dict = copy.deepcopy(kwargs_dict)
    for k,v in default_args.items():
        if not (k in kwargs_dict):
            kwargs_dict[k] = v
    return kwargs_dict

def _get_kwargs(args,**defaults):
    args = args if exists(args) else dict()
    return set_default_kwargs(args,**defaults)

def get_nan_chuncks(r,return_type='slice'):
    """
    r      -> position of mouse numpy.prod(r,axis=1) where r is r_nose or r_center or r_tail
    return_type -> 'slice': returns a slice object; 'index': returns all indices; 'firstlast': returns first and last indices of each chunck as a tuple

    returns
        idx -> list with continuous slices of r that are nan; idx[0] -> first nan chunck in r; etc
    """
    if not(type(r) is numpy.ndarray):
        r = numpy.asarray(r)
    #f = numpy.isnan(r) #numpy.logical_and(numpy.isnan(r[:-1]),numpy.isnan(r[1:]))
    k_start = numpy.nonzero(numpy.logical_and( numpy.isnan(r[1:]) , numpy.logical_not(numpy.isnan(r[:-1])) ) )[0] + 1 # index of the start of an avalanche
    k_end = numpy.nonzero(numpy.logical_and( numpy.isnan(r[:-1]) , numpy.logical_not(numpy.isnan(r[1:]))   ) )[0] + 1 # index of the end of an avalanche
    if numpy.isnan(r[0]):
        k_start = numpy.insert(k_start,0,0)
    if numpy.isnan(r[-1]):
        k_end = numpy.insert(k_end,len(k_end),len(r))
    idx = []    
    if (len(k_start) > 0) and (len(k_end) > 0):
        if return_type.lower() == 'slice':
            idx = [ slice(a,b) for a,b in zip(k_start,k_end) ]
        elif return_type.lower() == 'index':
            idx = [ numpy.arange(a,b) for a,b in zip(k_start,k_end) ]
        else:
            idx = [ (a,b) for a,b in zip(k_start,k_end) ]
    return idx

def try_or_default(f,default=numpy.nan, msg=''):
    try:
        return f()
    except:
        if len(msg)>0:
            print(msg)
        return default

def is_list_of_structtype(x):
    return (type(x) is list) and (type(x[0]) is structtype)

def is_list_of_1d_collection_or_none(x,min_n_elem=0,collection_is_str=False,allow_none_element=True):
    if (type(x) is type(None)):
        return False
    if not(type(x) is list):
        return False
    return all( (is_valid_1d_collection(xx,min_n_elem) or (allow_none_element and (type(xx) is type(None))) or (collection_is_str and (type(xx) is str))) for xx in x)

def is_valid_1d_collection(x,min_n_elem=0):
    if type(x) is type(None):
        return False
    return ((type(x) == numpy.ndarray) and (x.ndim==1) and (x.size>=min_n_elem)) or ( ((type(x) is list) or (type(x) is tuple)) and (len(x)>=min_n_elem))

def get_size_func(x):
    if (type(x) is numpy.ndarray):
        if x.ndim > 1:
            return lambda xx: xx.shape[0]
        else:
            return lambda xx: xx.size
    else:
        return len

def get_element_or_none(x,k):
    """
    returns an element of x if k < len(x), otherwise returns None

    if x is not a collection, returns x if k == 0, otherwise returns None
    """
    if (type(x) is type(None)) or ((type(x) is str) and (len(x) == 0)):
        return None
    get_size = get_size_func(x)
    if numpy.isscalar(x):
        #if k == 0:
        return x
        #else:
        #    return None
    else:
        if k < get_size(x):
            return x[k]
        else:
            return None
    #if k == 0:
    #    return x[0] if (not numpy.isscalar(x)) else x
    #else:
    #    return x[k] if (not numpy.isscalar(x)) and (k < get_size(x)) else None

def select_from_list(x,select_func):
    """
    returns only the items x[i] in x in which select_func(x[i]) is true
    """
    assert type(x) is list,'select_from_list ::: x must be a list'
    return [ xx for xx in x if select_func(xx) ]

def flatten_list(items,only_lists=False,return_list=False):
    """Yield items from any nested iterable; see Reference."""
    if return_list:
        return list(flatten_list_generator(items,only_lists=only_lists))
    else:
        return flatten_list_generator(items,only_lists=only_lists)
        

def flatten_list_generator(items,only_lists=False):
    try:
        from collections.abc import Iterable                            # < py38
    except Exception:
        from typing import Iterable 
    if only_lists:
        check_x = lambda xx: isinstance(xx, list)
    else:
        check_x = lambda xx: isinstance(xx, Iterable) and (not isinstance(xx, (str, bytes)))
    if check_x(items):
        for x in items:
            if check_x(x):
                for sub_x in flatten_list_generator(x,only_lists=only_lists):
                    yield sub_x
            else:
                yield x
    else:
        yield items

def get_item_recurrently(k,lst):
    if numpy.isscalar(lst):
        lst = [lst]
    if (type(lst) is numpy.ndarray) and (lst.dim > 1):
        raise ValueError('lst must be a 1d array or a list, or a tuple')
    return lst[k % len(lst)]

def _get_zero_or_same(x):
    return x[0] if len(x) == 1 else x

def unpack_list_of_tuples(lst):
    """
    given a list of len m where element is an n-tuple, then returns an n-tuple where each element containins m elements
    (similar to numpy.transpose)

    this is useful to unpack a list comprehension applied to a function that has multiple returns
    """
    return tuple( list(x) for x in zip(*lst) )

def transpose_list_of_list(lst):
    return list(zip(*lst))

def unique_ordered(lst):
    return list(collections.OrderedDict.fromkeys(lst))

def unique_stable(a):
    if not(type(a) is numpy.ndarray):
        a = numpy.asarray(a)
    assert a.ndim == 1,"input must be a 1-dim numpy.ndarray"
    _, idx = numpy.unique(a, return_index=True)
    return a[numpy.sort(idx)]

def calc_lower_upper_mean(x):
    if not _is_numpy_array(x):
        x = numpy.asarray(x)
    if x.ndim > 1:
        x=x.flatten()
    x_mean = nanmean(x)
    return nanmean(x[x<x_mean]),nanmean(x[x>x_mean])

def remove_key(mydict,*key):
    for k in key:
        mydict.pop(k,None)
    return mydict

def sort_each_row(x,row_ind):
    """
    sort x[k,:] according to row_ind[k]
    such that x_new[k,:] = x[k,row_ind[k]]
    """
    assert x.ndim == 2,"sort_each_row :: x must be 2-dim array"
    assert x.shape[0] == len(row_ind),"sort_each_row :: you must have one row_ind for each row in x"
    x_new = x.copy()
    for k in range(x.shape[0]):
        if len(row_ind[k]) > 0:
            assert len(row_ind[k]) == x.shape[1],"sort_each_row :: len(row_ind[k]) must have the same number of elements of x columns"
            x_new[k,:] = x[k,row_ind[k]]
    return x_new

def _get_distribution_bin_edges(x,n_bins,binning):
    binning = binning.lower()
    assert binning in ['linear','log'],"binning must be either 'linear' or 'log'"
    x_min = nanmin(asarray_nanfill(x).flatten())
    x_max = nanmax(asarray_nanfill(x).flatten())
    if binning == 'linear':
        # the edges of the histogram, as defined by the numpy.histogram function
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
        x_edges = numpy.linspace(x_min, x_max, n_bins+1)
    else:
        # log-scaled bins
        x_edges = numpy.logspace(numpy.log(x_min+1.0), numpy.log(x_max+1.0), n_bins+1, base=numpy.exp(1))-1.0
    return x_edges

def calc_cumulative_dist(dist_struct,x_par_name='x',P_par_name='P'):
    """
    given a distribution struct returned by calc_distribution below,
    we return the cumulative distribution (sum up to x)
    """
    if type(dist_struct) is list:
        return [ calc_cumulative_dist(d,x_par_name=x_par_name,P_par_name=P_par_name) for d in dist_struct ]
    else:
        C    = numpy.cumsum(dist_struct[P_par_name])
        return structtype(struct_fields=[x_par_name,'C'],field_values=[dist_struct[x_par_name],C])

def calc_distribution(x,n_bins=25,x_edges=None,return_as_struct=False,binning='linear',join_samples=False,replace_Peq0_by_nan=False,remove_Peq0=False,_recalculate_mid=True):
    """
    calculates the distribution (histogram) of x

    if x is list, then calculates the average distribution of x, all within the same x edges

    returns:
        * x mid points
        * P(x)
        * P(x) std dev
        * P(x) std err
    """
    binning = binning.lower()
    assert binning in ['linear','log'],"binning must be either 'linear' or 'log'"
    if join_samples:
        x = asarray_nanfill(x).flatten()
    if binning == 'linear':
        calc_x_mid    =  lambda xxe: (xxe[1:] + xxe[:-1])/2.0 # average of adjacent edges
    else:
        calc_x_mid    =  lambda xxe: numpy.exp(  numpy.log((xxe[1:]+1.0)*(xxe[:-1]+1.0))/2.0    )-1.0 # average of the log of adjacent edges
    if replace_Peq0_by_nan:
        check_gt_zero = lambda x: numpy.logical_not(numpy.isnan(x))
    else:
        check_gt_zero = lambda x: x>0.0
    remove_Peq0_func     = lambda x,P,Psd,Pse: ( x[check_gt_zero(P)],P[check_gt_zero(P)],Psd[check_gt_zero(P)],Pse[check_gt_zero(P)] )
    normalize_to_density = lambda P,x_edg: P/(numpy.sum(P)*numpy.diff(x_edg))
    if type(x) is list:
        if not exists(x_edges):
            ## the edges of the histogram, as defined by the numpy.histogram function
            ## https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
            x_edges = _get_distribution_bin_edges(x,n_bins,binning)
        x_mid = calc_x_mid(x_edges)
        P_x = get_empty_list(len(x))
        for k,xx in enumerate(x):
            _,P_x[k],_,_ = calc_distribution(xx,n_bins=n_bins,x_edges=x_edges,replace_Peq0_by_nan=replace_Peq0_by_nan,remove_Peq0=False,_recalculate_mid=False)
        P_x_avg = nanmean( asarray_nanfill(P_x), axis=0) #average of P_x
        P_x_sd  = nanstd(  asarray_nanfill(P_x), axis=0) #standard deviation of P_x
        P_x_se  = nanserr( asarray_nanfill(P_x), axis=0) # / numpy.sqrt(len(P_x)) #standard error of P_x
        if remove_Peq0:
            if _recalculate_mid:
                ind_x_edges_to_remove = numpy.nonzero(numpy.logical_not(check_gt_zero(P_x_avg)))[0]+1
            x_mid,P_x_avg,P_x_sd,P_x_se = remove_Peq0_func(x_mid,P_x_avg,P_x_sd,P_x_se)
            if _recalculate_mid:
                x_edges = numpy.delete(x_edges,ind_x_edges_to_remove)
                x_mid   = calc_x_mid(x_edges)
                P_x_avg = normalize_to_density(P_x_avg,x_edges)
    else:
        if not exists(x_edges):
            ## the edges of the histogram, as defined by the numpy.histogram function
            ## https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
            x_edges = _get_distribution_bin_edges(x,n_bins,binning)
        x_mid = calc_x_mid(x_edges)
        P_x_avg,_     = numpy.histogram(x, bins=x_edges, density=True)
        P_x_sd,P_x_se = numpy.zeros(P_x_avg.size),numpy.zeros(P_x_avg.size)
        if replace_Peq0_by_nan:
            P_x_avg[P_x_avg==0.0] = numpy.nan
        if remove_Peq0:
            if _recalculate_mid:
                ind_x_edges_to_remove = numpy.nonzero(numpy.logical_not(check_gt_zero(P_x_avg)))[0]+1
            x_mid,P_x_avg,P_x_sd,P_x_se = remove_Peq0_func(x_mid,P_x_avg,P_x_sd,P_x_se)
            if _recalculate_mid:
                x_edges = numpy.delete(x_edges,ind_x_edges_to_remove)
                x_mid   = calc_x_mid(x_edges)
                P_x_avg = normalize_to_density(P_x_avg,x_edges)
    if return_as_struct:
        return structtype(x=x_mid,P=P_x_avg,Psd=P_x_sd,Pse=P_x_se)
    else:
        return x_mid,P_x_avg,P_x_sd,P_x_se

def calc_dispersion_rows(X,Y,return_as_struct=True,return_separated=False):
    """
    calculates dispersion between each row of X and Y
    matrices have N rows

    if return_as_struct:
        returns list of N structtype objects with fields: r_mean, r_cov, r_dispersion, r_eigdir
    if return_separated:
        returns the lists of len N (one item per row of X,Y): r_mean, r_cov, r_dispersion, r_eigdir
    otherwise:
        returns list of N tuple, each tuple for each row, containing (r_mean, r_cov, r_dispersion, r_eigdir)
    """
    assert _is_numpy_array(X) and _is_numpy_array(Y), "X and Y must be numpy arrays"
    assert X.shape == Y.shape, "X and Y must have the same shape"
    if return_as_struct:
        get_result = lambda D: structtype(struct_fields=('r_mean', 'r_cov', 'r_dispersion', 'r_eigdir'),field_values=D)
    elif return_separated:
        get_result = lambda D: D
    result = [ get_result(calc_dispersion(x=x,y=y))  for x,y in zip(X,Y) ]
    if return_separated:
        if return_as_struct:
            warnings.warn('calc_dispersion_rows:return_separated parameter is ignored -- returning as a list of structtype')
            return result
        r_mean, r_cov, r_dispersion, r_eigdir = unpack_list_of_tuples(result)
        return r_mean, r_cov, r_dispersion, r_eigdir
    else:
        return result

def calc_dispersion(r=None,r_count=None,x=None,y=None):
    """
    r        -> r[k,:] = (x,y)
    if r is not givem x and y must be given (1d numpy arrays of same size)
    r_count  -> weight for each x,y pair in r (assumed 1 for each pair if None)

    calculates the x,y dispersion coordinates (eigenvalues and eigenvectors of the x,y covariance matrix)

    returns
        r_mean        -> (x_mean,y_mean)
        r_cov         -> x,y covariance matrix
        r_dispersion  -> equivalent to standard deviation (sqrt(abs(eigenvalues)))
        r_eigdir      -> list of eigenvectors of cov matrix, giving the dispersion spatial profile
    """
    if not exists(r):
        assert exists(x) and exists(y), "x and y must be provided"
        r = numpy.array((x,y)).T
    if not exists(r_count):
        r_count = numpy.ones(r.shape[0],dtype=float)
    r_mean       = numpy.sum(r*r_count[:,numpy.newaxis],axis=0) / numpy.sum(r_count)
    if (r_count.size == 1):
        r_count  = r_count[0]*numpy.ones(5)
        r        = numpy.tile(r[0,:], (5,1))
    r_cov        = numpy.cov(r, rowvar=False, fweights=r_count)
    l,v          = numpy.linalg.eig(r_cov)
    r_dispersion = numpy.sqrt(numpy.abs(l))
    r_eigdir     = [ v[:,m].flatten() for m in range(v.shape[1]) ]
    return r_mean, r_cov, r_dispersion, r_eigdir

def is_iterable(x):
    try:
        it = iter(x)
        return True
    except TypeError:
        return False

def intersect(a, b):
    a1, ia = numpy.unique(a, return_index=True)
    b1, ib = numpy.unique(b, return_index=True)
    aux    = numpy.concatenate((a1, b1))
    aux.sort()
    c      = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[numpy.isin(a1, c)], ib[numpy.isin(b1, c)]

def setdiff(A,B):
    r = []
    k = []
    for i,a in enumerate(A):
        if not (a in B):
            r.append(a)
            k.append(i)
    return numpy.asarray(r),numpy.asarray(k)

def factorize(num):
    k = int(copy.copy(num))
    return [n for n in range(1, k+1) if k%n == 0]

def divide_into_factors(num):
    s    = numpy.sqrt(num)
    if numpy.floor(s) == s:
        a = int(s)
        b = a
        return (a,b)
    f    = factorize(num)
    p    = numpy.cumprod(f)
    k    = find_first(f>s)
    a    = int(p[k-1])
    b    = int(num/a)
    return numpy.min((a,b)),numpy.max((a,b))

def is_list_of_list(v):
    if type(v) is list:
        if type(v[0]) is list:
            return True
    return False
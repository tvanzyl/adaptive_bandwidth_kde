# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:58:15 2016

@author: tvzyl

HANSEN LECTURE NOTES
http://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf


hall: says we should exclude points outside some boundary C in calculating getDensity
http://www.jstor.org/stable/2242395?seq=7#page_scan_tab_contents

|(x-X_i)/h_2|<C

h_i=h_2*f(X_i|h)^-0.5

where C >= c_1/c_2
K should vanish outside (-c_1, c_1)
f(x)^0.5 >= 2*c_2

c_1 = 2*sigma 
c_2 = mu^0.5 if we assume f(x) = mu

This indicates that performance in minimizing MISE, rather than ISE, should 
become the benchmark for measuring performance of bandwidth selection methods. 
http://link.springer.com/article/10.1007/BF01192160

"""
# cython: profile=True

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

import concurrent.futures

import numexpr as ne

import numpy as np
import pandas as pd

from numpy import zeros, dot, sqrt, exp, prod, sum, log, mean
from numpy import pi, tile, newaxis, isnan, isinf
from numpy import outer, atleast_2d, allclose
from numpy.linalg import det, inv
from math import factorial, gamma
from scipy.spatial.distance import cdist
from scipy.misc import factorial2
from scipy import optimize

from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.bandwidths import bw_scott, bw_silverman

from sklearn.base import BaseEstimator

from sys import float_info

from numba import jit, autojit, prange, types, typeof,generated_jit
import numba as nb

import psutil

def loglikelyhood(estimator, X, y):
    estimator.fit(X, y)    
    return log(estimator.predict(y)).mean()

def getLogLikelyHood(probabilities):
    return log(probabilities).mean()

#Constants
R_k_gaussian = 0.28209479177387814 #1/2/sqrt(pi)
k_2_gaussian = 1.0
eff_gaussian = 1.0513

def C_2_gaussian(q):
    r"""Gives
    
    .. math:: C_v(k,q) 
    from [1]_. Where :math:`v=2` and :math:`k` is gaussian
    
    References
    ----------
    .. [1] Hansen, B.E., 2009. Lecture notes on nonparametrics. Lecture notes.
    """
    v = 2
    c0 = 2. #factorial(v)
    c1 = 3. #factorial2(2*v-1, True)
    c2 = 1. #factorial2(v-1, True)
    numerator = pi**(q/2.) * 2.**(q+v-1.) * c0**2. * R_k_gaussian**q
    denominator =  v * k_2_gaussian**2. * ( c1 + (q-1.)*c2**2. )
    exponent = 1./(2.*v+q)
    return (numerator/denominator)**exponent
    

def getGlobalBandwidth( method, dataFrame, maxjobs=None):
    r"""
    Get Rule of thumb, Cross validation or Plug-in Bandwidth
    
    Returns estimated bandwidth as covariance matrix.
    
    We have no plug-in methods since statsmodels has droped plug-in 
    bandwidth selection methods because of their lack of robustness in a 
    multivariate setting.
    
    Parameters
    ----------
    method (str): 
        - cv_ml: cross validation maximum likelihood (statsmodels)
        - cv_ls: cross validation least squares (statsmodels)
        - normal_reference: Scott's normal reference rule of thumb (statsmodels)
        - silverman: Silverman's rule of thumb (scipy)
        - scott: Scott's rule of thumb (scipy)
        - over: oversmoothed upper bound [1]_
        - rule-of-thumb: multivariate rule-of-thumb [2]_
    Returns
    -------
    (h, H_diag, H) (ndarray, ndarray, ndarray):
        - h: is the bandwidth
        - H_diag: is the diagonal covariance matrix ie. h^2*I
        - H: is the full covariance matrix
    
    Examples
    --------
    dataFrame = pd.DataFrame(np.random.normal(size=(300,2)))
    for method in ['cv_ml','cv_ls','silverman','scott']:
        print(method, getGlobalBandwidth(method, dataFrame))
    
    References
    ----------
    .. [1] Hansen, B.E., 2009. Lecture notes on nonparametrics. Lecture notes.
    .. [2] Terrell, G.R., 1990. The maximal smoothing principle in density estimation. Journal of the American Statistical Association, 85(410), pp.470-477. http://www.jstor.org/stable/pdf/2289786.pdf?_=1465902314892
    
    """
    n, d = dataFrame.shape
    if method == 'cv_ls':        
        h = getCrossValidationLeastSquares(dataFrame, 1.0, bw_silverman(dataFrame).values, maxjobs=maxjobs)**0.5
    elif method == 'cv_ls_ndim':
        #rule-of-thumb
        h = dataFrame.std().values*C_2_gaussian(d)*n**(-1/(2.0*2.0+d))  
        H_diag = h**2
        H0 = outer(h,h)*dataFrame.corr()
        H = getCrossValidationLeastSquares(dataFrame, 1.0, H0.values, maxjobs=maxjobs)**0.5
    elif method in ['cv_ml','normal_reference']:
        var_type = 'c'*d
        dens_u = KDEMultivariate(data=dataFrame, var_type=var_type, bw=method)
        h = dens_u.bw
    elif method == 'silverman':
        h = bw_silverman(dataFrame).values
    elif method == 'scott':
        h = bw_scott(dataFrame).values
    elif method == 'over':
        h = dataFrame.std().values*( ( (d+8.)**((d+6.)/2.) * pi**(d/2.) * R_k_gaussian ) / (16 * n * gamma((d+8.)/2.) * (d+2.)) )**(1./(d+4.))        
    elif method == 'rule-of-thumb':
        h = dataFrame.std().values*C_2_gaussian(d)*n**(-1/(2.0*2.0+d))
    else:
        raise NotImplementedError(method)
    if method != 'cv_ls_ndim':
        H_diag = h**2
        H = outer(h,h)*dataFrame.corr().values
    return h, H_diag, H

class GlobalKDE(BaseEstimator):
    r"""
    
    References
    ----------
    [1] de Lima, M. S., & Atuncar, G. S. (2011). A Bayesian method to estimate the optimal bandwidth for multivariate kernel estimator. Journal of Nonparametric Statistics, 23(1), 137-148.
    """
    def __init__(self, method, covariance='H2'):
        self.method = method
        self.covariance = covariance
    def fit(self, X, y=None):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        """
        self.dataFrame_ = pd.DataFrame(X)
        self._pilot_H1, self._pilot_H2, self._pilot_H3 = getGlobalBandwidth(self.method, self.dataFrame_)
        if self.covariance == 'H2':
            self._pilot_H = self._pilot_H2
        elif self.covariance == 'H3':
            self._pilot_H = self._pilot_H3
        else:
            raise NotImplementedError(self.covariance)
        self.H_ = self._pilot_H
        return self
    def score(self, data):
        """Compute the mean log-likelihood under the model.
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        Returns
        -------
        logprob : float
            mean log-likelihood of the data in X.
        """
        return np.mean( np.log( getSamplePointDensity(self.dataFrame_, self.H_, pd.DataFrame(data)) ) )
    def predict(self, data):
        """Evaluate the density model on the data.
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).
        Returns
        -------
        density : ndarray, shape (n_samples,)
            The array of density evaluations.
        """
        return getSamplePointDensity(self.dataFrame_, self.H_, pd.DataFrame(data))

_mol_dist = lambda ds_i, allBW_i: dot(dot(ds_i, allBW_i), ds_i)

@jit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:], nb.float64[:,:]), 
     nopython=True, parallel=True, nogil=True)
def mahalanobisdist(XA, XB, VI):
    assert(XA.shape[1]==XB.shape[1])
    assert(VI.shape[0]==VI.shape[1])
    assert(VI.shape[1]==XA.shape[1])
    MA = XA.shape[0]
    MB = XB.shape[0]   
    N = XA.shape[1]    
    D = np.empty((MA, MB), dtype=np.float64)
    for i in prange(MA):
        for j in range(MB):
            d = 0.0
            for k in range(N):
                for l in range(N):                    
                    d += (XA[i, l] - XB[j, l])*VI[k,l]*(XA[i, k] - XB[j, k])
            D[i, j] = np.sqrt(d)
    return D

@jit(nb.float64[:](nb.int64, nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64), 
     nopython=True, nogil=True)
def _getSamplePointDensityWorkerFull(i, points, atPoint, inv_cov, log_sqrt_det_cov):
    _, d = points.shape
    #squared since the mahalanobis distance is a sqrt
    energy = mahalanobisdist(points, atPoint, inv_cov).flatten()**2.0    
    return exp( (-0.5*energy) - log((2.0*pi)**(0.5*d)) - log_sqrt_det_cov )

def getSamplePointDensityWorkerFull(args):
    r"""
    Note: we are evaluating for all points.
    
    .. math:: K(x_i, \mu, \Sigma) = \frac{1}{\sqrt{ (2\pi)^d |\Sigma| }} \exp\left(-\frac{1}{2} (\mu-x_i)\Sigma^{-1}(\mu-x_i) \right)
    """
    i, points, atPoint, inv_cov, log_sqrt_det_cov = args
    return _getSamplePointDensityWorkerFull(i, points, atPoint, inv_cov, log_sqrt_det_cov)

@jit(nb.float64[:](nb.int64, nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64), 
     nopython=True, nogil=True)
def _getSamplePointDensityWorkerDiag(i, points, atPoint, inv_cov, log_sqrt_det_cov):
    r"""
    Note: we are evaluating for all points.
    
    .. math:: K(x_i, \mu, \Sigma) = \frac{1}{\sqrt{ (2\pi)^d |\Sigma| }} \exp\left(-\frac{1}{2} (\mu-x_i)\Sigma^{-1}(\mu-x_i) \right)
    """
    _, d = points.shape
    ds = points - atPoint
    energy = sum(inv_cov*ds**2,axis=1)
    #return exp(-0.5*energy) / (2.0*pi)**(0.5*d) / sqrt_det_cov[i]    
    return exp( (-0.5*energy) -log((2.0*pi)**(0.5*d)) -log_sqrt_det_cov)

def getSamplePointDensityWorkerDiag(args):
    r"""
    Note: we are evaluating for all points.
    
    .. math:: K(x_i, \mu, \Sigma) = \frac{1}{\sqrt{ (2\pi)^d |\Sigma| }} \exp\left(-\frac{1}{2} (\mu-x_i)\Sigma^{-1}(\mu-x_i) \right)
    """
    i, points, atPoint, inv_cov, log_sqrt_det_cov = args
    return _getSamplePointDensityWorkerDiag(i, points, atPoint, inv_cov, log_sqrt_det_cov)

@jit(nb.float64[:](nb.int64, nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64), 
     nopython=True, nogil=True)
def _getGaussianConvolutionFull(i, points, atPoint, inv_cov, log_sqrt_det_cov):
    _, d = points.shape
    #squared since the cdist mahalanobis distance is a sqrt
    #Since this is a convolution with itself we omite the sum of mean
    energy = mahalanobisdist(points, atPoint, inv_cov).flatten()**2.0
    return exp( (-0.25*energy) - log((4.0*pi)**(0.5*d)) -log_sqrt_det_cov )

def getGaussianConvolutionFull(args):
    r"""
    Get the multivariate gaussian convolution kernel.
    Note: we are evaluating for all points.
    
    .. math:: K(x_i, \mu, \Sigma) = \frac{1}{\sqrt{ (4\pi)^d |\Sigma| }} \exp\left(-\frac{1}{4} (2\mu-x_i)\Sigma^{-1}(2\mu-x_i) \right)
    """
    i, points, atPoint, inv_cov, log_sqrt_det_cov = args
    return _getGaussianConvolutionFull(i, points, atPoint, inv_cov, log_sqrt_det_cov)

@jit(nb.float64[:](nb.int64, nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64), 
     nopython=True, nogil=True)
def _getGaussianConvolutionDiag(i, points, atPoint, inv_cov, log_sqrt_det_cov):
    _, d = points.shape
    #Since this is a convolution with itself we omite the sum of mean
    ds = points - atPoint #[i:i+1]
    energy = sum(inv_cov*ds**2.0,axis=1)
    return exp( (-0.25*energy) -log((4.0*pi)**(0.5*d)) - log_sqrt_det_cov )

def getGaussianConvolutionDiag(args):
    r"""    
    Get the multivariate gaussian convolution kernel where :math:`\Sigma` is a diagonal matrix. 
    Note: we are evaluating for all points.
    
    .. math:: 
        K(x_i, \mu, \Sigma) = \frac{1}{\sqrt{ (4\pi)^d |\Sigma| }} \exp\left(-\frac{1}{4} (2\mu-x_i)\Sigma^{-1}(2\mu-x_i) \right)
    """
    i, points, atPoint, inv_cov, log_sqrt_det_cov = args
    return _getGaussianConvolutionDiag(i, points, atPoint, inv_cov, log_sqrt_det_cov)

#Evaluate density given a dataset and set of bandwidths at the points
def getSamplePointDensity(dataFrame, cov, points, kernel='gaussian', maxjobs=None):
    r"""   
    Parameters
    ----------
    dataFrame:
        the training dataset.
    cov:  
        an array of covaraiance matrices, each item may be a single value in which case assumed symetric, or a diagnal matrice, or a full matrice. If only one item is given the assumed same sigma.        
    points:
        the points at which the KDE will be estimated    
    kernel: 
        - gaussian
        - gaussian_convolve
        
    """
    m, d = points.shape
    n, e = dataFrame.shape
    
    result = np.zeros((m,))
    if cov.shape[0] != n:
        if len(cov.shape) == 1:
            cov = np.tile(cov, (n,1))
        else:
            cov = np.tile(cov, (n,1, 1))
    Pooler = ThreadPool if maxjobs==1 else Pool
    
    points_values = points.values
    dataFrame_values = dataFrame.values
    
#    maxcores = psutil.cpu_count(logical=False)
#    maxjobs = maxcores if maxjobs is None else maxjobs
    if len(cov.shape) == 3:
        with Pooler(maxjobs) as pool:
            det_cov = det(cov)
#            det_cov = np.array(pool.map(det, cov))
            if (det_cov<=0).any():
                return np.nan
            inv_cov = inv(cov)
#            inv_cov = np.array(pool.map(inv, cov))
            log_sqrt_det_cov = log(sqrt(det_cov))
            # loop over dataFrame
            # Assumes that dataFrame < points
            if kernel=='gaussian':
                kernel = getSamplePointDensityWorkerFull
            elif kernel=='gaussian_convolve':
                kernel = getGaussianConvolutionFull
            it = pool.imap_unordered(kernel, 
                                     ((i, points_values, dataFrame_values[i:i+1], inv_cov[i], log_sqrt_det_cov[i]) for i in range(n)),
                                     chunksize=int(n/cpu_count()))
            for i in it: #range(n):
                result += i #getSamplePointDensityWorkerFull((i, points, dataFrame, inv_cov, sqrt_p_c))
    elif len(cov.shape) <= 2:
        with Pooler(maxjobs) as pool:
            det_cov = prod(cov, axis=1)
            inv_cov = 1./cov
            log_sqrt_det_cov = log(sqrt(det_cov))
            # loop over dataFrame 
            # Assumes that dataFrame < points
            if kernel=='gaussian':
                kernel = getSamplePointDensityWorkerDiag
            elif kernel=='gaussian_convolve':
                kernel = getGaussianConvolutionDiag                
            it = pool.imap_unordered(kernel, 
                                     [(i, points_values, dataFrame_values[i:i+1], inv_cov[i], log_sqrt_det_cov[i]) for i in range(n)],
                                     chunksize=int(n/cpu_count()))
            for i in it: #range(n):
                result += i #getSamplePointDensityWorkerDiag((i, points, dataFrame, inv_cov, sqrt_p_c))
                #ds = points - dataFrame.iloc[i]
                #energy = np.sum(inv_cov[i]*ds**2,axis=1)
                #result += exp(-0.5*energy) / sqrt_p_c[i]
    result /= n
    return result

@jit(nb.float64[:](nb.float64[:,:], nb.float64[:], nb.float64[:,:], nb.boolean), 
     nopython=True, parallel=True, nogil=True)
def getBalloonDensity_H1( samples, cov, points, notconvolved):
    """
    Variable Bandwidth Kernel Density Estimator.
    
    Parameters
    ----------
    samples: 
        the training dataset as a pandas samples
    cov: 
        an array of covariance matrices, each item may be a single value 
        in which case assumed symetric, or a diagnal matrice, or a full matrice.
        If only one item is given the assumed same sigma.
    points: 
        pandas samples with points at which the KDE will be estimated
    kernel:
        - gaussian
        - gaussain_convolve
    LOO:
        leave one out
    """
    m, d = points.shape
    n, e = samples.shape
    
    if notconvolved:
        c0 = (2.*pi)**(0.5*d)
        c1 = -0.50
    else:
        c0 = (4.*pi)**(0.5*d)
        c1 = -0.25
    result = zeros((m,), dtype=np.float64)
    
    ic_mod = m+1 if cov.shape[0] == m else 1
    
    # loop over points    
    for i in prange(m):
        CV = cov[i%ic_mod]
        det_cov = CV
        inv_cov = 1./CV
        sqrt_p_c = c0*np.sqrt(det_cov)    
        ds = points[i] - samples
        energy = np.sum(inv_cov*(ds**2),axis=1)
        result[i] = np.sum(np.exp(c1*energy) / sqrt_p_c)
    return result/n

@jit(nb.float64[:](nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.boolean), 
     nopython=True, parallel=True, nogil=True)
def getBalloonDensity_H2( samples, cov, points, notconvolved):
    """
    Variable Bandwidth Kernel Density Estimator.
    
    Parameters
    ----------
    samples: 
        the training dataset as a pandas samples
    cov: 
        an array of covariance matrices, each item may be a single value 
        in which case assumed symetric, or a diagnal matrice, or a full matrice.
        If only one item is given the assumed same sigma.
    points: 
        pandas samples with points at which the KDE will be estimated
    kernel:
        - gaussian
        - gaussain_convolve
    LOO:
        leave one out
    """
    m, d = points.shape
    n, e = samples.shape
    
    if notconvolved:
        c0 = (2.*pi)**(0.5*d)
        c1 = -0.50
    else:
        c0 = (4.*pi)**(0.5*d)
        c1 = -0.25
    result = zeros((m,), dtype=np.float64)
    
    ic_mod = m+1 if cov.shape[0] == m else 1
    
    # loop over points    
    for i in prange(m):
        CV = cov[i%ic_mod]
        det_cov = np.prod(CV)
        inv_cov = 1./CV
        sqrt_p_c = c0*np.sqrt(det_cov)
        ds = points[i] - samples
        energy = np.sum(inv_cov*(ds**2), axis=1)        
        result[i] = np.sum(np.exp(c1*energy) / sqrt_p_c)
        print(energy.shape)
    return result/n

@jit(nb.float64[:](nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], nb.boolean), 
     nopython=True, parallel=True, nogil=True)
def getBalloonDensity_H3( samples, cov, points, notconvolved ):
    """
    Variable Bandwidth Kernel Density Estimator.
    
    Parameters
    ----------
    samples: 
        the training dataset as a pandas samples
    cov: 
        an array of covariance matrices, each item may be a single value 
        in which case assumed symetric, or a diagnal matrice, or a full matrice.
        If only one item is given the assumed same sigma.
    points: 
        pandas samples with points at which the KDE will be estimated
    kernel:
        - gaussian
        - gaussain_convolve
    LOO:
        leave one out
    """    
    m, d = points.shape
    n, e = samples.shape    
    
    if notconvolved:
        c0 = (2.*pi)**(0.5*d)
        c1 = -0.50
    else:
        c0 = (4.*pi)**(0.5*d)
        c1 = -0.25
    result = zeros((m,), dtype=np.float64)
    
    ic_mod = m+1 if cov.shape[0] == m else 1
    
    # loop over points
    for i in prange(m):
        CV = cov[i%ic_mod]
        det_cov = np.linalg.det(CV)
        inv_cov = np.linalg.inv(CV)
        sqrt_p_c = c0*np.sqrt(det_cov)
        energy = mahalanobisdist(points[i:i+1], samples, inv_cov).flatten()**2
        result[i] = np.sum(np.exp(c1*energy) / sqrt_p_c)
    return result/n

@generated_jit(nopython=True, parallel=True, nogil=True)
def getBalloonDensity(samples, cov, points, notconvolved):
    if cov == types.Array(types.float64, 1, 'C'):
        return getBalloonDensity_H1
    elif cov == types.Array(types.float64, 2, 'C'):
        return getBalloonDensity_H2
    elif cov == types.Array(types.float64, 3, 'C'):
        return getBalloonDensity_H3
    else:
        raise ValueError('unsuported covariance')

@jit(nb.float64[:](nb.float64[:,:], nb.float64[:,:,:]),
     nopython=True, nogil=True, parallel=True)
def getLOODensity_H3(samples, cov):
    r"""
    .. math:: \hat{f}(X,H) = \sum_{i\in X}\sum_{j\in X; i\neq j}{K(x_i-x_j, H_j)}
    
    Parameters
    ----------
    dataFrame:
        the training dataset.
    cov:  
        an array of covariance matrices, each item may be a scalar value in which case assumed symetric, or a diagnal matrix, or a full matrix.
    """    
    n, _ = samples.shape
    
    ic_mod = n+1 if cov.shape[0] == n else 1
    
    result = np.zeros((n,))
    
    for i in prange(n):
        CV = cov[i%ic_mod]
        det_cov = det(CV)
        inv_cov = inv(CV)
        log_sqrt_det_cov = log(sqrt(det_cov))
        it = _getSamplePointDensityWorkerFull(i, samples, samples[i:i+1], inv_cov, log_sqrt_det_cov)
        it[i] = 0
        result += it #getSamplePointDensityWorkerFull((i, points, dataFrame, inv_cov, sqrt_p_c))
    result /= n-1
    return result

@jit(nb.float64[:](nb.float64[:,:], nb.float64[:,:]),
     nopython=True, nogil=True, parallel=True)
def getLOODensity_H2(samples, cov):
    r"""
    .. math:: \hat{f}(X,H) = \sum_{i\in X}\sum_{j\in X; i\neq j}{K(x_i-x_j, H_j)}
    
    Parameters
    ----------
    dataFrame:
        the training dataset.
    cov:  
        an array of covariance matrices, each item may be a scalar value in which case assumed symetric, or a diagnal matrix, or a full matrix.
    """    
    n, _ = samples.shape
    
    ic_mod = n+1 if cov.shape[0] == n else 1
    
    result = np.zeros((n,))
    
    for i in prange(n):
        CV = cov[i%ic_mod]
        det_cov = prod(CV)
        inv_cov = 1./CV
        log_sqrt_det_cov = log(sqrt(det_cov))
        it = _getSamplePointDensityWorkerDiag(i, samples, samples[i:i+1], inv_cov, log_sqrt_det_cov)
        it[i] = 0 #getSamplePointDensityWorkerDiag((i, points, dataFrame, inv_cov, sqrt_p_c))
        result += it
    result /= n-1
    return result

def getLOODensity(samples, cov, maxjobs=None ):
    r"""
    .. math:: \hat{f}(X,H) = \sum_{i\in X}\sum_{j\in X; i\neq j}{K(x_i-x_j, H_j)}
    
    Parameters
    ----------
    dataFrame:
        the training dataset.
    cov:  
        an array of covariance matrices, each item may be a scalar value in which case assumed symetric, or a diagnal matrix, or a full matrix.
    """    
    n, e = samples.shape
    
    result = np.zeros((n,))
    if cov.shape[0] != n:
        if len(cov.shape) == 1:
            cov = np.tile(cov, (n,1))
        else:
            cov = np.tile(cov, (n,1, 1))
    Pooler = ThreadPool if maxjobs==1 else Pool
    
    if len(cov.shape) == 3:
        with Pooler(maxjobs) as pool:
#            f_old = np.array(pool.map(det, cov))
            det_cov = det(cov)
            if (det_cov<=0).any():
                return np.nan
#            f_old = np.array(pool.map(inv, cov))            
            inv_cov = inv(cov)
            log_sqrt_det_cov = log(sqrt(det_cov))
            # loop over dataFrame
            # Assumes that dataFrame < points
            it = pool.imap(getSamplePointDensityWorkerFull, 
                           ((i, samples, samples[i:i+1], inv_cov[i], log_sqrt_det_cov[i]) for i in range(n)),
                           chunksize=int(n/cpu_count()))
            for j, i in enumerate(it): #range(n):
                i[j] = 0
                result += i #getSamplePointDensityWorkerFull((i, points, dataFrame, inv_cov, sqrt_p_c))
    elif len(cov.shape) <= 2:        
        with Pooler(maxjobs) as pool:
            det_cov = prod(cov, axis=1)
            inv_cov = 1./cov
            log_sqrt_det_cov = log(sqrt(det_cov))
            it = pool.imap(getSamplePointDensityWorkerDiag, 
                           ((i, samples, samples[i:i+1], inv_cov[i], log_sqrt_det_cov[i]) for i in range(n)),
                           chunksize=int(n/cpu_count()))
            for j, i in enumerate(it): #range(n):
                i[j] = 0 #getSamplePointDensityWorkerDiag((i, points, dataFrame, inv_cov, sqrt_p_c))
                result += i
    result /= n-1
    return result

def getIMSE_H1_H2( alpha, cov, dataFrame, covariance_class, d, iu, maxjobs=None ):
    r"""    
    :math:`\mathcal{H}_1=\{\alpha h^2\mathbf{I}\}`
    
    :math:`\mathcal{H}_2=\{\alpha\  \mathrm{diag}(h_1^2,\dots,h_d^2)\}`
    
    :math:`\mathcal{H}_3=\{\alpha \mathbf{\Sigma}\}`
    
    Parameters
    ----------
    h:
        :math:`\sqrt{h}`
    dataFrame:
        dataFrame, :math:`X`
    cov:
        the covariance matrix, :math:`H`
    
    Returns
    -------
    IMSE:
        .. math:: \frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{N}
            \bar{K}_{H_j}(X_{i},X_{j})-\frac{2}{n(n-1)}\sum_{i=1}^{n}
            \sum_{j=1,j\neq i}^{N}K_{H_j}(X_{i},X_{j})
    
    Where :math:`\bar{K}_{h}` is the multivariate convolution kernel    
    """
    alpha = alpha**2.
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        parallel1 = executor.submit(getSamplePointDensity, dataFrame, cov*alpha, dataFrame, kernel='gaussian_convolve', maxjobs=maxjobs)
        parallel2 = executor.submit(getLOODensity, dataFrame.values, cov*alpha, maxjobs=maxjobs)
    result = mean(parallel1.result()) - 2*mean(parallel2.result())
    return float_info.max if isnan(result) or isinf(result) else result


def getIMSE_H3( alpha, cov, dataFrame, covariance_class, d, iu, maxjobs=None ):
    r"""    
    :math:`\mathcal{H}_1=\{\alpha h^2\mathbf{I}\}`
    
    :math:`\mathcal{H}_2=\{\alpha\  \mathrm{diag}(h_1^2,\dots,h_d^2)\}`
    
    :math:`\mathcal{H}_3=\{\alpha \mathbf{\Sigma}\}`
    
    Parameters
    ----------
    h:
        :math:`\sqrt{h}`
    dataFrame:
        dataFrame, :math:`X`
    cov:
        the covariance matrix, :math:`H`
    
    Returns
    -------
    IMSE:
        .. math:: \frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{N}
            \bar{K}_{H_j}(X_{i},X_{j})-\frac{2}{n(n-1)}\sum_{i=1}^{n}
            \sum_{j=1,j\neq i}^{N}K_{H_j}(X_{i},X_{j})
    
    Where :math:`\bar{K}_{h}` is the multivariate convolution kernel    
    """
    #Unrolling function for performance
    #alpha = rollSigma( alpha,d,iu )
    #rollSigma(res, d, iu=None)
    #if iu is None:
    #    iu = triu_indices(d, 1)
    res=alpha
    d0 = res[:d]
    c0 = outer(d0, d0)
    p0 = res[d:]
    rho = ones((d,d))
    rho[iu] = p0
    rho[(iu[1], iu[0])] = p0
    alpha = c0*rho
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        parallel1 = executor.submit(getSamplePointDensity, dataFrame, cov*alpha, dataFrame, kernel='gaussian_convolve', maxjobs=maxjobs)
        parallel2 = executor.submit(getLOODensity, dataFrame.values, cov*alpha, maxjobs=maxjobs)
    result = mean(parallel1.result()) - 2*mean(parallel2.result())
    return float_info.max if isnan(result) or isinf(result) else result

from numpy import triu_indices, diag, r_, ones

def getCrossValidationLeastSquares( dataFrame, lambdas, h0=1.0, maxjobs=None):
    r"""
    Obtain an estimate for cross validated least squares as per [1]_ pg.106.
    Given a fixed :math:`\lambda_j` estimate the global parameter :math:`h`.
    
    Parameters
    ----------
    dataFrame:
        The training data
    lambdas:
        The fixed covariance(s) or :math:`\lambda_j`.
    h0:
        Initial guess at :math:`\sqrt{h_{opt}}`.
    Returns
    -------
    :math:`\sqrt{h_{opt}}`
    
    Reference
    ---------        
    .. [1] Silverman, B.W., 1986. Density estimation for statistics and data analysis (Vol. 26). CRC press.
    .. [2] proposed by Rudemo (1982) and Bowman (1984),
    """    
    try:
        #if it was a full covariance matrix make it an array of lambdas
        if len(lambdas.shape) == 2 and lambdas.shape[0] == lambdas.shape[1] and allclose(lambdas.T, lambdas):
            lambdas = lambdas[newaxis]
        else:
            pass #was an array of lambdas either 2dim or 3dim
    except AttributeError: #must be a number
        lambdas = np.asarray([lambdas])
    d = dataFrame.shape[1]
    iu = triu_indices(d, 1)
    try:
        if len(h0.shape) == 2:
            covariance_class='H3'            
            #Square covariance matrix            
            h0 = unrollSigma( h0,d,iu )
            getIMSE = getIMSE_H3
        elif h0.shape[0] > 1:
            covariance_class='H2'
            getIMSE = getIMSE_H1_H2
        else:
            covariance_class='H1'
            getIMSE = getIMSE_H1_H2
    except AttributeError:
        covariance_class='H1' #h0 is a number not an array
        getIMSE = getIMSE_H1_H2
    res = optimize.minimize(getIMSE, x0=h0, 
                            args=(lambdas, dataFrame, covariance_class, d, iu, maxjobs), 
                            method='Nelder-Mead',   #115sec, 0.0123554                          
                            tol=0.04,
#                            method='BFGS',          #264sec, 0.012418
#                            tol=0.018,
                            )
    if covariance_class=='H3':
        #This is the full covariance matrix        
        h = rollSigma( res.x,d,iu )
    else:
        # This is the square root, either a diag or a scalar
        h = res.x**2
    return h

def unrollSigma(H0, d, iu=None):
    r"""
    Unroll a covariance matrix into standard deviations and corelations cofficents
    
    Parameters
    ----------
        H0:
            A Covariance matrix
        iu:
            Upper triangle, less diagonal indices for covariance matrix
    """
    if iu is None:        
        iu = triu_indices(d, 1)
    d0 = diag(H0)**0.5
    c0 = outer(d0, d0)
    p0 = H0/c0
    return r_[d0,  p0[iu]]

def rollSigma(res, d, iu=None):
    r"""
    Oposite of unrollSigma. See unrollSigma.
    """
    if iu is None:        
        iu = triu_indices(d, 1)
    d0 = res[:d]
    c0 = outer(d0, d0)
    p0 = res[d:]
    rho = ones((d,d))
    rho[iu] = p0
    rho[(iu[1], iu[0])] = p0
    return c0*rho
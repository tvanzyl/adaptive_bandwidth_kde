# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:57:38 2016

@author: tvzyl
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:29:29 2012

Adapted form the same file written by Christian van de Walt
@author: tvzyl

Algorithm from
http://www.prasa.org/proceedings/2010/prasa2010-04.pdf

ML-LOO Maximum Likelyhood Leave One Out
http://www.sciencedirect.com/science/article/pii/S0167865512001948
getLOODensity
http://dspace.nwu.ac.za/bitstream/handle/10394/10635/VanDerWalt_CM.pdf?sequence=1

regularizers:
spherical
diagonal
squared_euclid

"""
import numpy as np
import pandas as pd

from numpy import zeros, mean, newaxis, dot, prod, square, exp, pi, sum
from numpy.linalg import norm
from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator

import numba as nb
from numba import jit, prange

import mvn

@jit(nopython=True, nogil=True, parallel=True)
def mvnpdf(x, mu, cov):
    det_cov = prod(cov)
    if det_cov == 0:
        return np.zeros((1,))
    d = x.shape[1]
    ds = x - mu
    inv_cov = 1/cov
    desc = dot(square(ds), inv_cov.T)
    c0 = (2.*pi)**(-0.5*d)
    c1 = det_cov**-0.5
    return c0*c1*exp(-0.5*desc)

def getBandwidths(dataFrame, cov, regularizer, maxit, eps, eps_d, dist):
    nvec, ndim = dataFrame.shape
    if eps_d == 'vdwalt':
        #Introduced by vdwalt as a minium distance
        if dist is None:
            kdt = KDTree(dataFrame, leaf_size=30, metric='euclidean')
            dist, nn = kdt.query(dataFrame, k=2, return_distance=True)
        lowerbound = np.empty(nvec)
#        lowerbound[:] = mean(dist[:,1]**2.0)  #Leiva et al. 2012
        lowerbound[:] = (dist[:,1]**2.0)/ndim #vdwalt
    else:
        lowerbound = np.empty(nvec)
        lowerbound[:] = eps_d
    if regularizer == 'vdwalt':
        return getBandwidths_vdwalt(dataFrame.values, cov, maxit, eps, lowerbound)
    elif  regularizer == 'barnard':
        return getBandwidths_barnard(dataFrame.values, cov, maxit, eps, lowerbound)
    elif  regularizer == 'leiva':
        return getBandwidths_barnard(dataFrame.values, cov, maxit, eps, lowerbound)
    else:
        raise NotImplementedError()

@jit(nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.int64, nb.float64, nb.float64[:]),
     nopython=True, nogil=True, parallel=True)
def getBandwidths_vdwalt(samples, cov, maxit, eps, lowerbound):
    r"""
    parameters
    ----------
    regularizer: str
        one of:
        - barnard: [1]_
        - vdwalt:
        - leiva: 
    
    .. math:: CV=\frac{1}{n^{2}}
    
    References
    ----------
    .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice. Princeton University Press. (2007)
    """
    nvec, ndim = samples.shape
    allBW = np.empty((nvec, ndim), dtype=np.float64)
    allBW[:] = cov
#    allBW[:] = mean(cov)
    xi = samples    
    converged = np.empty((nvec,), dtype=np.int64)
    converged[:] = 0
    
    for iit in range(maxit):        
        allBW_old = allBW.copy()
        phi = mvn.getLOODensity_H2(samples, allBW)
        converged[:] = 0
        for k in prange(nvec):
            xk = samples[k]
            ds = (xk-xi)**2.0
            kHk = mvnpdf(xi, xk, allBW_old[k])
            kHk[k] = 0.0
            kHk_phi = kHk/phi
            dbw = 0.0
            for l in range(ndim):
                allBW[k, l] = max(np.sum(ds[:,l]*kHk_phi)/np.sum(kHk_phi), lowerbound[k])
                dbw += (allBW[k, l] - allBW_old[k, l])**2.0
            if dbw**0.5 < eps:
                converged[k] = 1
        if np.sum(converged) == nvec:
            break
    if np.sum(converged) == nvec:
        print( "converged ", (iit+1) )
    else:
        print( "max iterations reached" )
    return allBW


@jit(nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.int64, nb.float64, nb.float64[:]),
     nopython=True, nogil=True, parallel=True)
def getBandwidths_barnard(samples, cov, maxit, eps, lowerbound):
    r"""
    parameters
    ----------
    regularizer: str
        one of:
        - barnard: [1]_
        - vdwalt:
        - leiva: 
    
    .. math:: CV=\frac{1}{n^{2}}
    
    References
    ----------
    .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice. Princeton University Press. (2007)
    """
    nvec, ndim = samples.shape
    allBW = np.empty((nvec, ndim), dtype=np.float64)
    allBW[:] = cov
#    allBW[:] = mean(cov)
    xi = samples    
    converged = np.empty((nvec,), dtype=np.int64)
    converged[:] = 0
    
    for iit in range(maxit):        
        allBW_old = allBW.copy()        
        converged[:] = 0
        for k in prange(nvec):
            xk = samples[k]
            ds = (xk-xi)**2.0
            kHk = mvnpdf(xi, xk, allBW_old[k])
            kHk[k] = 0.0            
            dbw = 0.0
            for l in range(ndim):
                allBW[k, l] = max(np.sum(ds[:,l]*kHk)/np.sum(kHk), lowerbound[k])
                dbw += (allBW[k, l] - allBW_old[k, l])**2.0
            if dbw**0.5 < eps:
                converged[k] = 1
        if np.sum(converged) == nvec:
            break
    if np.sum(converged) == nvec:
        print( "converged ", (iit+1) )
    else:
        print( "max iterations reached" )
    return allBW

@jit(nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.int64, nb.float64, nb.float64[:]),
     nopython=True, nogil=True, parallel=True)
def getBandwidths_leiva(samples, cov, maxit, eps, lowerbound):
    r"""
    parameters
    ----------
    regularizer: str
        one of:
        - barnard: [1]_
        - vdwalt:
        - leiva: 
    
    .. math:: CV=\frac{1}{n^{2}}
    
    References
    ----------
    .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice. Princeton University Press. (2007)
    """
    nvec, ndim = samples.shape
    allBW = np.empty((nvec, ndim), dtype=np.float64)
    allBW[:] = cov
#    allBW[:] = mean(cov)
    xi = samples    
    converged = np.empty((nvec,), dtype=np.int64)
    converged[:] = 0
    
    for iit in range(maxit):        
        allBW_old = allBW.copy()
        phi = mvn.getLOODensity_H2(samples, allBW)
        converged[:] = 0
        for k in prange(nvec):
            xk = samples[k]
            ds = (xk-xi)**2.0
            kHk = mvnpdf(xi, xk, allBW_old[k])
            kHk[k] = 0.0            
            dbw = 0.0
            for l in range(ndim):
                allBW[k, l] = max(1/(ndim*nvec**2-ndim*nvec)*sum(ds[:, l]*kHk, axis=0)*sum(1./phi, axis=0), lowerbound[k])
                dbw += (allBW[k, l] - allBW_old[k, l])**2.0
            if dbw**0.5 < eps:
                converged[k] = 1
        if np.sum(converged) == nvec:
            break
    if np.sum(converged) == nvec:
        print( "converged ", (iit+1) )
    else:
        print( "max iterations reached" )
    return allBW

#def getBandwidths_leiva(dataFrame, cov, regularizer='vdwalt', maxit=1, eps=1e-6, eps_d='vdwalt'):
#    r"""
#    parameters
#    ----------
#    regularizer: str
#        one of:
#        - barnard: [1]_
#        - vdwalt:
#        - leiva: 
#    
#    .. math:: CV=\frac{1}{n^{2}}
#    
#    References
#    ----------
#    .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice. Princeton University Press. (2007)
#    """
#    nvec, ndim = dataFrame.shape
#    allBW = zeros((nvec,ndim))
#    dbw = zeros(nvec)
#    if regularizer=='leiva':
#        allBW[:] = mean(cov)
#    else:
#        allBW[:] = cov
#    maxit_reached = False
#    if eps_d == 'vdwalt':
#        #Introduced by vdwalt as a minium distance
#        kdt = KDTree(dataFrame, leaf_size=30, metric='euclidean')
#        dist, nn = kdt.query(dataFrame, k=2, return_distance=True)
#        lowerbound = (dist[:,1]**2)/ndim
#    else:
#        lowerbound = zeros(nvec)
#        lowerbound[:] = eps_d
#    xi = dataFrame.values
#    for iit in range(maxit):
#        if regularizer in ['vdwalt', 'leiva']:
#            phi = mvn.getLOODensity(dataFrame, allBW)[newaxis].T
#        for k in range(nvec):
#            Hk = allBW[k]
#            xk = dataFrame.iloc[k].values
#            ds = (xk-xi)**2.
#            kHk = mvnpdf(xi, xk, Hk)[newaxis].T
#            kHk[k] = 0
#            if regularizer=='vdwalt':
#                newBW = sum(ds*kHk/phi, axis=0)/sum(kHk/phi, axis=0)
#            elif regularizer=='barnard':
#                newBW = sum(ds*kHk, axis=0)/sum(kHk, axis=0)
#            elif regularizer=='leiva':
#                newBW = 1/(ndim*nvec**2-ndim*nvec)*sum(ds*kHk, axis=0)*sum(1./phi, axis=0)
#                #newBW = sum(ds*kHk, axis=0)/sum(phi, axis=0)
#            else:
#                raise NotImplementedError(regularizer)
#            #have we converged to close to zero then keep our previous value
#            if (newBW < lowerbound[k]).any():
#                newBW[newBW < lowerbound[k]] = lowerbound[k]
#            dbw[k]=norm(newBW-Hk)
#            Hk[:] = newBW
#        if ( dbw < eps).all():
#            break
#    if iit>=maxit-1:
#        maxit_reached=True
#    if maxit_reached:
#        print( "max iterations reached" )
#    return allBW

class LikelihoodKDE(BaseEstimator):
    r"""
    
    References
    ----------
    """
    def __init__(self, H0='cv_ls', regularizer='vdwalt', eps_d='vdwalt', maxit=10):
        self.H0 = H0
        self.regularizer = regularizer
        self.eps_d = eps_d
        self.maxit = maxit
        self.dist_ = None        
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
        if isinstance(self.H0, str):
            self._pilot_h, self._pilot_H_diag, self._pilot_H = mvn.getGlobalBandwidth(self.H0, self.dataFrame_)
        else:
            self._pilot_H_diag = self.H0
        self.H_ = getBandwidths(self.dataFrame_, self._pilot_H_diag, 
                                self.regularizer, self.maxit, 1e-3, 
                                self.eps_d, self.dist_)
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
        return np.mean( np.log( mvn.getSamplePointDensity(self.dataFrame_, self.H_, pd.DataFrame(data)) ) )
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
        return mvn.getSamplePointDensity(self.dataFrame_, self.H_, pd.DataFrame(data))
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:02:29 2016

@author: tvzyl
"""
from scipy.spatial.distance import cdist
from numpy import sum, array, newaxis, ndarray, outer, fromiter
from numpy.linalg import det, inv
import pandas as pd
import numpy as np
import numexpr as ne
import numpy as np

import mvn
from sklearn.base import BaseEstimator

from multiprocessing import Pool, cpu_count
from numba import jit, prange, generated_jit
import numba as nb

@jit(nb.float64[:,:](nb.float64[:], nb.float64[:,:], nb.float64[:,:], nb.int64, nb.int64),
     nopython=True)
def worker_entropy(point, dataFrame, V, n, p):
    d = n**(2.0/(p+4.0))+p
    ds = point-dataFrame          
    result = np.zeros((p,p), dtype=np.float64)
    omega_norm = 0.0
    for i in range(n):        
        delta = outer(ds[i],ds[i]) + V
        det_delta_pow = det(delta)**(-(d+1.0)/2.0)
        omega_norm += det_delta_pow
        result += det_delta_pow*inv(delta)
    return inv(result/omega_norm)/(d+1.0)

@jit(nb.float64[:,:](nb.float64[:], nb.float64[:,:], nb.float64[:,:], nb.int64, nb.int64),
     nopython=True)
def worker_quadratic(point, dataFrame, V, n, p):
    d = n**(2.0/(p+4.0))+p
    ds = point-dataFrame
    result = np.zeros((p,p), dtype=np.float64)
    omega_norm=0.0
    for i in range(n):        
        delta = outer(ds[i],ds[i]) + V
        det_delta_pow = det(delta)**(-(d+1.0)/2.0)
        omega_norm += det_delta_pow
        result += delta*det_delta_pow
    return result/(d+1.0)/omega_norm

def getDensity(dataFrame, points, bayes='lima_entropy'):
    n, p = dataFrame.shape
    m, _ = points.shape
    V = dataFrame.cov().values    
    if bayes=='lima_entropy':
        worker = worker_entropy
    elif bayes=='lima_quadratic':
        worker = worker_quadratic
    else:
        raise NotImplementedError(bayes)
    with Pool() as pool:
        hk = np.array(pool.starmap(worker, 
                                   ((points.values[j], dataFrame.values, V, n, p) for j in range(m)),
                                   chunksize=int(m/cpu_count())))
    return mvn.getBalloonDensity(dataFrame.values, hk, points.values, True)

class BayesianKDE(BaseEstimator):
    r"""
    
    References
    ----------
    [1] de Lima, M. S., & Atuncar, G. S. (2011). A Bayesian method to estimate the optimal bandwidth for multivariate kernel estimator. Journal of Nonparametric Statistics, 23(1), 137-148.
    """
    def __init__(self, bayes='lima_entropy'):
        self.bayes = bayes
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
        return np.mean( np.log( getDensity(self.dataFrame_, pd.DataFrame(data), self.bayes) ) )
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
        return getDensity(self.dataFrame_, pd.DataFrame(data), self.bayes)

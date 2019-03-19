# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:17:44 2016

@author: tvzyl
"""
import pandas as pd
from sklearn.utils.extmath import cartesian

import os
from numpy import vstack,repeat,digitize,linspace,zeros, ndarray
from numpy import mean,log,sqrt,arange, ones, outer, atleast_2d
from numpy import triu_indices, isnan, asarray, diag, r_, tril_indices

import mvn
from sklearn.base import BaseEstimator
from math import floor
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from scipy import optimize

import sys

import samplepoint

def getBinnedUnbiasedIMSE( cov, dataFrame, bin_means, bin_mean, covariance_class, d, iu, selection):
    r"""    
    :math:`\mathcal{H}_1=\{h^2\mathbf{I}\}`
    
    :math:`\mathcal{H}_2=\{\mathrm{diag}(h_1^2,\dots,h_d^2)\}`
    
    :math:`\mathcal{H}_3=\{\mathbf{\Sigma}\}`
    
    Parameters
    ----------
    dataFrame:
        dataFrame, :math:`X`
    cov:
        the covariance matrix, :math:`H`
    bin_mean:
        the mean of the bin being optimised
    
    Returns
    -------
    IMSE:
        .. math:: \frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{N}
            \bar{K}_{H_j}(X_{i},X_{j})-\frac{2}{n(n-1)}\sum_{i=1}^{n}
            \sum_{j=1,j\neq i}^{N}K_{H_j}(X_{i},X_{j})
    
    Where :math:`\bar{K}_{h}` is the multivariate convolution kernel    
    """    
    if covariance_class=='H3':        
        cov = mvn.rollSigma( cov, d, iu )
    elif covariance_class=='H2':
        cov = cov**2.
    else: #covariance_class=='H1':
        cov = ones(d)*cov**2.
    nj = selection.sum()
    n = selection.shape[0]
    Rf = mvn.getSamplePointDensity(dataFrame, cov, dataFrame, kernel='gaussian_convolve', maxjobs=1).mean()
    Ks = mvn.getSamplePointDensity(bin_mean, cov, dataFrame, maxjobs=1)
    f_1 = 2.0*( Ks[selection].sum() + Ks[-selection].sum() )/(n-1)
    #print(cov, Rf, f_1)
    #assert False        
    return sys.float_info.max if isnan(f_1) else Rf - f_1

def worker(args):
    p, processors, partitions, dataFrame, H0, covariance_class, n, d = args
    list_bins = []
    list_means = []
    list_digitized = []
    for k in range(d):
        partition = partitions[k]
        data = dataFrame[k]
        #min, max, parts count
        dim_bins = linspace(partition[0], partition[1], partition[2]+1)
        list_means.append( (dim_bins[:-1]+dim_bins[1:])/2. )
        bin_dig = digitize(data, dim_bins)
        bin_dig[bin_dig==partition[2]+1]=partition[2]
        list_digitized.append( bin_dig )
        list_bins.append(linspace(1, partition[2], partition[2]))
    digitized = vstack(list_digitized).T
    bins = cartesian(list_bins)
    bin_means = cartesian(list_means)
    selections = []
    H_s = []
    #calculate this processors chunk of bins
    chunks = arange(len(bins))%processors == p
    iu = triu_indices(d, 1)
    if covariance_class=='H3':
        #Square covariance matrix
        h0 = mvn.unrollSigma(H0, iu)
    else:
        h0 = H0**0.5
    for bin, amean in zip(bins[chunks], bin_means[chunks]):
        selection = (digitized==bin).all(axis=1)
        if selection.any():
            res = optimize.minimize(getBinnedUnbiasedIMSE, x0=h0, args=(dataFrame, bin_means, pd.DataFrame(atleast_2d(amean)), covariance_class, d, iu, selection), method='BFGS', options={'gtol':1e-4,'eps':1e-5})            
            res = res.x
            if covariance_class=='H3':                
                H_s.append( mvn.rollSigma(res, d, iu) )
            elif covariance_class=='H2':
                H_s.append( res**2. )
            else:
                H_s.append( ones(d)*res**2. )
            selections.append( selection )
    return selections, H_s

def getBandwidths(dataFrame, covariance_class='H3', partitions='droot'):
    n, d = dataFrame.shape
    if covariance_class == 'H1':        
        _, GH_, _ = mvn.getGlobalBandwidth('rule-of-thumb', dataFrame)        
        GH_ = GH_.mean()
    elif covariance_class=='H2':
        _, GH_, _ = mvn.getGlobalBandwidth('rule-of-thumb', dataFrame)
    elif covariance_class=='H3':        
        _, _, GH_ = mvn.getGlobalBandwidth('rule-of-thumb', dataFrame)    
    else:
        raise NotImplementedError('Unknown Covariance Class')
    allBW = ndarray((n,)+GH_.shape)
    allBW[:] = GH_
    if partitions=='lit':
        if   d==2:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(151, d) ]).T
        elif d==3:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(51, d) ]).T
        elif d==4:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(21, d) ]).T
        elif d==5:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(11, d) ]).T
        elif d==6:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(7, d) ]).T            
        elif d==7:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(4, d) ]).T            
        elif d<=9:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(3, d) ]).T
        else:
            partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(2, d) ]).T
    elif partitions=='droot':
        partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(floor(n**(1/d)), d) ]).T
    elif isinstance(partitions, (int, float, complex)):
        partitions = vstack([ dataFrame.min().values, dataFrame.max().values, repeat(partitions, d) ]).T
    #spkde = samplepoint.SamplePointKDE(covariance=covariance)
    #spkde.fit(dataFrame)
    #cov_pilot = spkde.H_
    with Pool() as pool:
        processors = os.cpu_count()
        it = pool.imap_unordered( worker, ((p, processors, partitions, dataFrame, GH_, covariance_class, n, d) for p in range(processors)) )
        for selections, H_diags in it:
            for selection, H_diag in zip(selections, H_diags):
                print(allBW.shape, H_diag.shape, selection.shape, selection.sum())
                allBW[selection] = H_diag
    return allBW

class PartitionKDE(BaseEstimator):
    r"""
    
    References
    ----------
    [1] Sain, S.R., 2002. Multivariate locally adaptive density estimation. Computational Statistics & Data Analysis, 39(2), pp.165-186.    
    """
    def __init__(self, covariance='H3', partitions='droot'):
        self.partitions = partitions
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
        self.H_ = getBandwidths(pd.DataFrame(X), self.covariance, self.partitions)
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
        return mean( log( mvn.getSamplePointDensity(self.dataFrame_, self.H_, pd.DataFrame(data)) ) )
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

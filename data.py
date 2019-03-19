# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:11:51 2016

@author: tvzyl
"""

import numpy as np
import pandas as pd
from numpy import linalg as la

from numpy.linalg import det, inv
from scipy.stats import multivariate_normal, norm
from math import factorial
from numpy import ones, sum, ndarray, array, pi, dot, sqrt, newaxis, exp
from sklearn.utils.extmath import cartesian
from sklearn.base import BaseEstimator

from ellipsoidpy import Sk

class TrueBlobDensity(BaseEstimator):
    def __init__(self, means, covs, ratios=None):
        self.means = means
        self.covs = covs
        self.ratios = ratios
    def fit(self, X=None, y=None):
        self.norms_ = [multivariate_normal(mean=mean, cov=cov) for mean, cov in zip(self.means,self.covs)]
        return self
    def predict(self, data):
        if self.ratios is not None:
            return np.sum( np.c_[[ratio*norm.pdf(data) for norm, ratio in zip(self.norms_,self.ratios)]], axis=0)
        else:
            return np.mean(np.c_[[norm.pdf(data) for norm in self.norms_]], axis=0)        
    def getSampleSizes(self, n_samples, n_folds):
        ranges = ndarray((n_folds))
        ranges[0:int(n_samples%n_folds)] = n_samples//n_folds + 1 
        ranges[int(n_samples%n_folds):] = n_samples//n_folds
        return ranges        
    def getIntegratedSquaredPDF(self):
        result = 0
        for fi, fj in cartesian([np.arange(3),np.arange(3)]):
            sigma_sum = self.covs[fi]+self.covs[fj]
            inv_sigma_sum = inv(sigma_sum)
            det_sigma_sum = det(sigma_sum)
            mu_diff = (self.means[fi] - self.means[fj])[newaxis]
            normalising_const = sqrt(2*pi*det_sigma_sum)*exp(-0.5*dot(dot(mu_diff,inv_sigma_sum), mu_diff.T))
            result += self.ratios[fi]*self.ratios[fj]/normalising_const
        return result
    def sample(self, n_samples=1, random_state=None, withclasses=False):
        n_folds, d = self.means.shape
        if self.ratios is not None:
            sizes = (n_samples*ones((n_folds))*self.ratios).astype(int)
            sizes[-1] += n_samples-sum(sizes)
        else:
            sizes = self.getSampleSizes(n_samples, n_folds)
        samples = ndarray((int(n_samples), int(d)))
        classes = ndarray((int(d)))
        start = 0
        for i in range(n_folds):        
            end = start+int(sizes[i])
            samples[start:end] = np.random.multivariate_normal( self.means[i], self.covs[i], size=int(sizes[i]) )
            classes[start:end] = i
            start=end
        if withclasses:
            return samples, classes
        else:
            return samples

class TrueBallDensity(BaseEstimator):
    def __init__(self, mean, cov, inner_trials=10):
        self.mean = array(mean)        
        self.cov = cov    
        self.inner_trials = inner_trials
        self.dimensions_ = self.mean.shape[0]
    def fit(self, X=None, y=None):
        self.a_ = multivariate_normal(mean=self.mean, cov=self.cov)
        self.b_ = multivariate_normal(mean=self.mean, cov=self.cov)
        return self
    def predict(self, data):
        return self.normal_fact()**-1. * self.a_.pdf(data)*(1.-self.b_.pdf(data))**self.inner_trials
    def normal_fact(self):
        #https://en.wikipedia.org/wiki/Triangular_number
        #https://en.wikipedia.org/wiki/Tetrahedral_number
        tri_num = lambda n,c: factorial(n)/factorial(n-c)/factorial(c)
        po = self.inner_trials+1
        cov = self.cov
        k = self.dimensions_
        return sum((1 if term%2==0 else -1)*tri_num(po,term)/sqrt(term+1)/sqrt((2*pi)**k*det(cov))**term for term in range(0,po+1))
    def getIntegratedSquaredPDF(self):
        raise NotImplemented
    def sample(self, n_samples=1, random_state=None, withclasses=False):
        if withclasses:
            raise NotImplementedError("withclasses")
        c_samples = 0
        s = ndarray((n_samples,self.dimensions_))
        try:
            while c_samples < n_samples:
                u = np.random.uniform(size=n_samples)                
                y = self.a_.rvs(n_samples)
                tmp_samples = y[u < (1-self.b_.pdf(y))**self.inner_trials/10.0 ]
                c_tmp_samples = tmp_samples.shape[0]
                s[c_samples:c_samples+c_tmp_samples] = tmp_samples
                c_samples += c_tmp_samples
        except ValueError:
            s[c_samples:] = tmp_samples[:n_samples-c_samples]
        return s

class TrueEllipsoidDensity(BaseEstimator):
    def __init__(self, radii, var):
        self.radii_ = array(radii)        
        self.dimensions_ = self.radii_.shape[0]
        self.var_ = var
    def fit(self, X=None, y=None):        
        return self
    def predict(self, data):        
        radii_ = self.radii_
        var_ = self.var_  
        x2 = np.dot(data, np.diag(1./radii_) )
        r2 = la.norm(x2, axis=1)
        e2 = np.dot(1./r2[:,np.newaxis]*x2, np.diag(radii_))
        v2 = la.norm(e2, axis=1)
        u2 = la.norm(e2-data, axis=1)
        p = np.array([ norm.pdf(j, loc=0, scale=i*var_) for i, j in zip(v2,u2) ])
        return p*(1./Sk(np.ones(1), self.dimensions_))
    def getIntegratedSquaredPDF(self):
        raise NotImplemented
    def sample(self, n_samples=1, random_state=None, withclasses=False):
        if withclasses:
            raise NotImplementedError("withclasses")        
        dimensions_ = self.dimensions_
        radii_ = self.radii_
        var_ = self.var_        
        x = np.random.normal(size=(n_samples,dimensions_))
        r = la.norm(x, axis=1)
        e = np.dot(1./r[:,np.newaxis]*x, np.diag(radii_))
        v = la.norm(e, axis=1)
        u = np.array([norm.rvs(loc=0.,scale=i*var_) for i in v])
        s = e + e/la.norm(e, axis=1)[:,np.newaxis]*u[:,np.newaxis]
        return s

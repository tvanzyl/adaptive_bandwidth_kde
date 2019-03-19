# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:13:45 2016

@author: tvzyl
"""
#import pyximport; pyximport.install()
from balloon_cython import getDensity

from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

class BalloonKDE(BaseEstimator):
    r"""Get a balloon density.
    
    Can use any of a number of balloon density methods from the literature.
    Depends on f: pilot but not on number of samples or K pg(119) [1]
    www.tandfonline.com/doi/pdf/10.1080/02331880108802727
    
    Parameters
    ----------  
    dataFrame (DataFrame(n,d)):
        training data
    k (str or int):
        k-nearest neighbours rule or a value, options:
        
        - sqrt: Square root of n rule
        - kung: [1]
        - hansen: [2]
    points (DataFrame(m,d)):
        points at which we want the estimate the density
    balloon:
        - loftsgaarden-kernel-knn: [3] the smoothed kernel version
        - loftsgaarden-knn: [3]
        - terrel-knn: [4]
        - mittal: [5]        
    References
    ---------
    .. [1] Kung et al. (2012) http://www.sciencedirect.com/science/article/pii/S0167715212001927?np=y
    .. [2] Hansen http://www.ssc.wisc.edu/~bhansen/718/NonParametrics10.pdf
    .. [3] Loftsgaarden et al. (1965) http://projecteuclid.org/download/pdf_1/euclid.aoms/1177700079
    .. [4] Terrell et al. (1992) pg.1258 http://www.jstor.org/stable/pdf/2242011.pdf
    .. [5] Mittal et al. (2004) http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=1315179
    .. [6] Hall et al. (1989) http://www.sciencedirect.com/science/article/pii/0167715289900023 
    """
    def __init__(self, k=1, balloon=None, percentile=0.6826):
        if balloon not in ['loftsgaarden-knn',
                           'loftsgaarden-kernel-knn',
                           'terrel-knn',                           
                           'hall-pilot-knn',
                           'kung-knn',
                           'biau-knn',
                           'biau-ellipse-knn',
                           'mittal']: raise NotImplementedError(balloon)
        self.k=k
        self.balloon = balloon
        self.percentile = percentile
        self.tree_ = None 
        self.dist_ = None
        self.nn_ = None
        self.dist_loo_ = None
        self.nn_loo_ = None
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
        self.tree_ = KDTree(self.dataFrame_, leaf_size=30, metric='euclidean')
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
        return np.mean( np.log( getDensity(self.dataFrame_, self.k, pd.DataFrame(data), self.balloon, self.tree_, percentile=self.percentile) ) )
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
        return getDensity(self.dataFrame_, 
                          self.k, pd.DataFrame(data), 
                          self.balloon, 
                          self.tree_,  
                          self.dist_,
                          self.nn_,
                          self.dist_loo_,
                          self.nn_loo_,
                          percentile=self.percentile)


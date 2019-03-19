# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:21:32 2016

@author: tvzyl

"""
import pyximport; pyximport.install()
from samplepoint_cython import getBandwidths
import numpy as np
import pandas as pd
import mvn

from sklearn.base import BaseEstimator

class SamplePointKDE( BaseEstimator):
    r"""Get a samplepoint kernel density estimate.        
    
    h_0 = pilot bandwidth 0
    
    s(x_i) = pilotfactor which may be the geometric mean of f gm(f(X))
    
    f(x|h_0) = pilot KDE function
    
    .. math:: h_i = h_0 \left(\frac{s(x_i)}{f(x_i|h_0)}\right)^\alpha
    
    iterate
    
    .. math:: h_i[0] = h_0(s(x)/f(x_i|h_0))^\alpha
    .. math:: h_i[j+1] = h_i[j](s(x)/f(x_i|h_i[j]))^\alpha
    
    Parameters
    ----------
    HO (str or ndarray(ndim,ndim), ndarray(ndim,)):            
        - see mvn.getGlobalBandwidth for options
    pilotfactor : {'silverman', 'breiman'}
        - silverman : :math:`s(x_i) = \prod( f(x_i) )^{1/n}` [1]_
        - breiman : :math:`s(x_i) = ||\mathrm{knn}(x_i)-x_i||`
        - wu : [9]_
    alpha:
        - '1/d': Terrell et al. (1992)
            equivalient axiomatic k(nn)
        - 0.5: Abrahamson (1982), Silverman (2000)        
    maxit : (int)
        maximum number of iterations for iterative optimiser
    f (): 
        - mvn.getLOODensity: a pilot kernel density estimate function
        - breiman: use the k nearest neighbours

    covariance : {'H1', 'H2', 'H3'}
        - H1 : :math:`\{h^2\mathbf{I}\}`
        - H2 : :math:`\{\mathrm{diag}(h_1^2,\dots,h_d^2)\}`
        - H3 : :math:`\{\mathbf{\Sigma}\}`

    optimser : (str)
        - substitute : [1] 101
        - cv_ls : [1] 105
        - iterate : 
    
    Attributes
    ----------
    H0_:
    dataFrame_:
    
    Notes
    -----
    Samplepoint Algorithms :
        - terrell : Terrell et al. (1992)
            alpha = '1/d';
            pilotfactor='silverman';
            h_i=h_0*(gm(f(x|h_0))/f(x_i|h_0))^(1/d)
        - breiman : Breiman et al. (1977)
            alpha = -1.0;
            pilotfactor=1.0
            f = 'breiman'
            k = k
            h_i=h_0*|k-nn(x_i)-x_i|;
        - silverman : Silverman (1986) [1]_ pg.101
            alpha=0.5
            pilotfactor='silverman'
            h_i=h_0*(gm(f(x|h_0))/f(x_i|h_0))^0.5
        - hall : Hall et al. (1988)
            alpha=1.0;
            pilotfactor = 1.0;
            h_1=h_0*(1/f(h_0))^0.5 as a reasonable h_1;
            h_i=h_1*(1/f(h_0))^0.5;
            h_i=h_0*(1/f(h_0))^1.0
        - abrahamson : Abrahamson (1982)
            alpha = 0.5;
            pilotfactor = 1;
            h_i=h_0*(1/f(h_0))^0.5

    See Salgado-Ugarte (2003), Izenman pg(5) and Van Kerm (2003) for an overview.
    
    Iteration added no value to the global kernel density estimate:
        Fox 1990 according to Salgado-Ugarte (2003) and Hall et al. (1988)
    
    kopt may be equal to d*integral(f^2); lim d->M: Terrell et al. (1992) pg(1259)

    References
    ----------
    .. [1] Silverman, B.W., 1986. Density estimation for statistics and data analysis (Vol. 26). CRC press.
        http://w0.ned.ipac.caltech.edu/level5/March02/Silverman/paper.ps
    Hall et al. (1988):
        http://link.springer.com/article/10.1007/BF00348751
    Abrahamson (1982):
        http://www.jstor.org/stable/2240724?seq=1#page_scan_tab_contents
    Breiman et al. (1977):
        http://www.jstor.org/stable/pdf/1268623.pdf?_=1459364945550
    Terrell et al. (1992):
        http://www.jstor.org/stable/pdf/2242011.pdf
    Salgado-Ugarte (2003):
        http://ageconsearch.umn.edu/bitstream/116063/2/sjart_st0036.pdf
    Izenman:
        http://astro.temple.edu/~alan/MMST/NPDE.PDF
    Van Kerm (2003):
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.467.9087&rep=rep1&type=pdf
    .. [9] Wu et al. (2007) A variable bandwidth selector in multivariate kernel density estimation         
        http://www.stat.ncku.edu.tw/faculty_private/tjwu/varbandnew.pdf
    """        

    def __init__(self, H0='rule-of-thumb', pilotfactor='silverman', alpha=0.5, covariance='H2', optimiser='substitute', maxit=3, k=1):
        self.H0 = H0
        self.pilotfactor = pilotfactor
        self.alpha = alpha
        self.optimiser = optimiser
        self.maxit=maxit
        self.k = k
        self.covariance = covariance
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
            self._pilot_H1, self._pilot_H2, self._pilot_H3 = mvn.getGlobalBandwidth(self.H0, self.dataFrame_)
            if self.covariance == 'H2':
                self._pilot_H = self._pilot_H2
            elif self.covariance == 'H3':
                self._pilot_H = self._pilot_H3
            else:
                raise NotImplementedError(self.covariance)
        else:
            self._pilot_H = self.H0
        if self.alpha == '1/d':
            self.alpha = 1./self.dataFrame_.shape[1]
        elif self.alpha == 'd':
            self.alpha = self.dataFrame_.shape[1]
        self.H_ = getBandwidths(self.dataFrame_, self._pilot_H, 
                                self.alpha, self.pilotfactor, 
                                self.optimiser, self.maxit, 
                                self.k, self.dist_)
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

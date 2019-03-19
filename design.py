# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:59:19 2016

@author: tvzyl
"""

import data

from sklearn.datasets import make_spd_matrix

from pandas import DataFrame
from numpy import mean, diag, eye, rot90, dot, array, abs
from numpy import zeros, ones, arange
from numpy.random import uniform

from scipy.stats import entropy

from sklearn.utils.extmath import cartesian

from sklearn.neighbors import KDTree

class Experiment():
    def __init__(self, n_training, n_test, dimensions, actualsEstimator, name):
        self.__name__ = name
        self.n_training = n_training
        self.n_test = n_test        
        self.dimensions = dimensions
        self.actualsEstimator = actualsEstimator.fit()
        self.train = DataFrame( self.actualsEstimator.sample(self.n_training) )        
        self.importance_test = DataFrame( self.actualsEstimator.sample(self.n_test) )        
        self.lows = self.train.min(axis=0)
        self.highs = self.train.max(axis=0)
        self.uniform_test  = DataFrame( uniform(low=self.lows, high=self.highs, size=(self.n_test,self.dimensions)) )        
        self.importance_actuals = self.actualsEstimator.predict(self.importance_test)
        self.uniform_actuals = self.actualsEstimator.predict(self.uniform_test)
        
        self.test = self.importance_test
        self.test_actuals = self.importance_actuals
        
        #Build up a KDTRee for faster processing
        self.kdt_ = KDTree(self.train, leaf_size=30, metric='euclidean')
        self.dist_, self.nn_ = self.kdt_.query(self.test, k=int(1+self.n_test**0.5), return_distance=True)
        self.dist_loo_, self.nn_loo_ = self.kdt_.query(self.train, k=int(1+self.n_training**0.5), return_distance=True)
        
        
    def ISE(self, estimates, actuals):
        r"""
        .. math:: Q_N(e,a,p) = \frac{1}{N}\sum_{i=0}^N\frac{(e_i-a_i)^2}{p_i} 
        Integrated Squared Error with Importance Sampling
        """
        return mean(((estimates - actuals)**2.))**0.5
        
    def IAE(self, estimates, actuals):
        r"""
        .. math:: Q_N(e,a,p) = \frac{1}{N}\sum_{i=0}^N\frac{|e_i-a_i|}{p_i} 
        Integrated Absolute Error with Importance Sampling
        """
        return mean(abs(estimates - actuals))

    def EmpericalEntropy(self, estimates):
        return entropy(estimates, base=2)

    def JensenShannon(self, estimates, actuals):
        M = 0.5*(estimates+actuals)
        return 0.5*(entropy(estimates,M, base=2) + entropy(actuals,M, base=2))
    
    def KullbackLeiber(self, estimates, actuals):
        return entropy(actuals, estimates, base=2)

    def getResults(self, estimator, prekdt=False):
#        uni_est = estimator.predict(self.uniform_test)    
        #Attach some  pre calculated results to the estimator
        if prekdt:
            estimator.nn_ = self.nn_
            estimator.dist_ = self.dist_
            estimator.nn_loo_ = self.nn_loo_
            estimator.dist_loo_ = self.dist_loo_
            estimator.kdt_ = self.kdt_

        est = estimator.predict(self.test) 
        actuals = self.test_actuals

        estimator.nn_ = None
        estimator.dist_ = None
        estimator.nn_loo_ = None
        estimator.dist_loo_ = None
        estimator.kdt_ = None
        
        return self.ISE(est, actuals), self.IAE(est, actuals), self.JensenShannon(est, actuals), self.KullbackLeiber(est, actuals), self.EmpericalEntropy(est)

class BlobExperiment(Experiment):
    def __init__(self, n_training, n_test, means, covs, ratios=None, name='BlobExperiment'):
        self.means = array(means)
        self.covs = array(covs)
        self.ratios = ratios
        dimensions = self.means.shape[1]
        actualsEstimator = data.TrueBlobDensity(self.means, self.covs, self.ratios).fit()
        Experiment.__init__(self, n_training, n_test, dimensions, actualsEstimator, name=name)
    def getSIGMA(self, std1,std2,rho):
        return dot(array([[std1,std2]]).T,array([[std1,std2]])) * (rot90(diag([rho,rho]))+eye(2))
    def getSIGMA_N(self, std,rho):    
        n=len(std)
        return dot(array([std]).T,array([std])) * (rho*(1-eye(n))+eye(n))

class BallExperiment(Experiment):
    def __init__(self, n_training, n_test, mean=[0,0], cov=0.5, inner_trials=10, name='ballExperiment'):
        self.mean = array(mean)
        self.cov = cov
        self.inner_trials = inner_trials
        dimensions = self.mean.shape[0]
        actualsEstimator = data.TrueBallDensity(self.mean, self.cov, self.inner_trials).fit()        
        Experiment.__init__(self, n_training, n_test, dimensions, actualsEstimator,  name=name)

class EllipsoidExperiment(Experiment):
    def __init__(self, n_training, n_test, radii=[3,2,1], cov=0.05, name='ballExperiment'):
        self.radii = array(radii)
        self.cov = cov
        dimensions = self.radii.shape[0]
        actualsEstimator = data.TrueEllipsoidDensity(self.radii, self.cov).fit()
        Experiment.__init__(self, n_training, n_test, dimensions, actualsEstimator, name=name)
    
class Experiment1(BlobExperiment):
    r"""
    Bimodal J
    Density H, A [1, 2].
    
    References
    ----------
    [1] Wand, M.P. and Jones, M.C., 1993. Comparison of smoothing parameterizations in bivariate kernel density estimation. Journal of the American Statistical Association, 88(422), pp.520-528. http://www.jstor.org/stable/pdf/2290332.pdf
    [2] Zhang, X., King, M.L. and Hyndman, R.J., 2006. A Bayesian approach to bandwidth selection for multivariate kernel density estimation. Computational Statistics & Data Analysis, 50(11), pp.3009-3031.
    """
    def __init__(self, n_training=200, n_test=2000):
        self.dimensions=2
        self.modes=2
        mu_1 = [ 2.0, 2.0]
        mu_2 = [-1.5,-1.5]
        sigma_1 = [[1.0,-0.9],[-0.9,1.0]]
        sigma_2 = [[1.0, 0.3],[ 0.3,1.0]]
        BlobExperiment.__init__(self, n_training, n_test, [mu_1,mu_2], [sigma_1,sigma_2], 
                            None, "e1_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,2))

class Experiment2(BlobExperiment):
    r"""
    Skewed
    Density C, #2,  [1, 2, 3].
    
    References
    ----------
    [1] Wand, M.P. and Jones, M.C., 1993. Comparison of smoothing parameterizations in bivariate kernel density estimation. Journal of the American Statistical Association, 88(422), pp.520-528. http://www.jstor.org/stable/pdf/2290332.pdf
    [2] Wu, T.J., Chen, C.F. and Chen, H.Y., 2007. A variable bandwidth selector in multivariate kernel density estimation. Statistics & probability letters, 77(4), pp.462-467. http://www.stat.ncku.edu.tw/faculty_private/tjwu/varbandnew.pdf
    [3] Sain, S.R., 2002. Multivariate locally adaptive density estimation. Computational Statistics & Data Analysis, 39(2), pp.165-186. http://private.igf.edu.pl/~jnn/Literatura_tematu/Sain_2002.pdf
    """
    def __init__(self, n_training=200, n_test=2000):
        self.dimensions=2
        self.modes=3
        mu_1 = [0.0, 0.0]
        mu_2 = [0.5, 0.5]
        mu_3 = [13./12.,13/12.]
        sigma_1 = diag([1, 1])
        sigma_2 = diag([2./3.,2./3.])**2
        sigma_3 = diag([5./9.,5./9.])**2
        ratios = [0.2,0.2,0.6]
        BlobExperiment.__init__(self, n_training, n_test, [mu_1,mu_2, mu_3], [sigma_1,sigma_2,sigma_3], 
                            ratios, "e2_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,3))

class Experiment3(BlobExperiment):
    r"""
    Trimodal Gavel
    Density I, D, F4, #7 [1,2,3,4].
    
    References
    ----------
    [1] Wand, M.P. and Jones, M.C., 1993. Comparison of smoothing parameterizations in bivariate kernel density estimation. Journal of the American Statistical Association, 88(422), pp.520-528. http://www.jstor.org/stable/pdf/2290332.pdf
    [2] de Lima, M.S. and Atuncar, G.S., 2011. A Bayesian method to estimate the optimal bandwidth for multivariate kernel estimator. Journal of Nonparametric Statistics, 23(1), pp.137-148. http://www.tandfonline.com/doi/pdf/10.1080/10485252.2010.485200
    [3] Zougab, N., Adjabi, S. and Kokonendji, C.C., 2014. Bayesian estimation of adaptive bandwidth matrices in multivariate kernel density estimation. Computational Statistics & Data Analysis, 75, pp.28-38. http://www.sciencedirect.com/science/article/pii/S0167947314000322
    [4] Wu, T.J., Chen, C.F. and Chen, H.Y., 2007. A variable bandwidth selector in multivariate kernel density estimation. Statistics & probability letters, 77(4), pp.462-467. http://www.stat.ncku.edu.tw/faculty_private/tjwu/varbandnew.pdf
    """
    def __init__(self, n_training=200, n_test=2000):
        self.dimensions = 2
        self.modes=3
        mu_1 = [-6./5., 6./5.]
        mu_2 = [6./5., -6./5.]
        mu_3 = [0.,0.]
        sigma_1 = self.getSIGMA(3./5.,3./5.,3./10.)
        sigma_2 = self.getSIGMA(3./5.,3./5.,-3./5.)
        sigma_3 = self.getSIGMA(1./4.,1./4.,1./5.)
        ratios = [9./20.,9./20.,2./20.]
        BlobExperiment.__init__(self, n_training, n_test, [mu_1,mu_2, mu_3], [sigma_1,sigma_2,sigma_3], 
                            ratios, "e3_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,3))

class Experiment4(BlobExperiment):
    r"""
    Trimodal :math:`\lambda`
    
    Density K, ,F3  [1,2,3].    
    
    References
    ----------
    [1] Wand, M.P. and Jones, M.C., 1993. Comparison of smoothing parameterizations in bivariate kernel density estimation. Journal of the American Statistical Association, 88(422), pp.520-528. http://www.jstor.org/stable/pdf/2290332.pdf
    [2] Sain, S.R., 2002. Multivariate locally adaptive density estimation. Computational Statistics & Data Analysis, 39(2), pp.165-186. http://private.igf.edu.pl/~jnn/Literatura_tematu/Sain_2002.pdf
    [3] Zougab, N., Adjabi, S. and Kokonendji, C.C., 2014. Bayesian estimation of adaptive bandwidth matrices in multivariate kernel density estimation. Computational Statistics & Data Analysis, 75, pp.28-38. http://www.sciencedirect.com/science/article/pii/S0167947314000322
    """    
    def __init__(self, n_training=200, n_test=2000):
        self.dimensions = 2
        self.modes=3
        mu_1 = [-1.,  0.]
        mu_2 = [ 1.,  3**0.5*2./3.]
        mu_3 = [ 1., -3**0.5*2./3.]
        sigma_1 = self.getSIGMA(3./5.,7./10.,3./5.)
        sigma_2 = self.getSIGMA(3./5.,7./10.,0. )
        sigma_3 = self.getSIGMA(3./5.,7./10.,0.)
        ratios = [3./7.,3./7.,1./7.]
        BlobExperiment.__init__(self, n_training, n_test, [mu_1,mu_2, mu_3], [sigma_1,sigma_2,sigma_3], 
                            ratios, "e4_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,3))

class Experiment5(BlobExperiment):
    r"""
    Bimodal T
    Density F2, C [1,2].
    
    References
    ----------
    [1] Zougab, N., Adjabi, S. and Kokonendji, C.C., 2014. Bayesian estimation of adaptive bandwidth matrices in multivariate kernel density estimation. Computational Statistics & Data Analysis, 75, pp.28-38. http://www.sciencedirect.com/science/article/pii/S0167947314000322
    [2] de Lima, M.S. and Atuncar, G.S., 2011. A Bayesian method to estimate the optimal bandwidth for multivariate kernel estimator. Journal of Nonparametric Statistics, 23(1), pp.137-148. http://www.tandfonline.com/doi/pdf/10.1080/10485252.2010.485200
    """
    def __init__(self, n_training=200, n_test=2000):
        self.dimensions=2
        self.modes=2
        mu_1 = [ 1.,  1.]
        mu_2 = [-1., -1.]
        sigma_1 = self.getSIGMA(1.,1., 0.5)
        sigma_2 = self.getSIGMA(1.,1.,-0.5)
        BlobExperiment.__init__(self, n_training, n_test, [mu_1,mu_2], [sigma_1,sigma_2], 
                            None, "e5_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,2))

class Experiment6(BlobExperiment):
    r"""
    Unimodal ND
    Density F5, F7, _, _ [1,2].
    
    References
    ----------
    [1] Zougab, N., Adjabi, S. and Kokonendji, C.C., 2014. Bayesian estimation of adaptive bandwidth matrices in multivariate kernel density estimation. Computational Statistics & Data Analysis, 75, pp.28-38. http://www.sciencedirect.com/science/article/pii/S0167947314000322
    [2] Duong, T. and Hazelton, M.L., 2005. Cross‐validation Bandwidth Matrices for Multivariate Kernel Density Estimation. Scandinavian Journal of Statistics, 32(3), pp.485-506. http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9469.2005.00445.x/pdf
    """
    def __init__(self, dimensions=3, n_training=300, n_test=2000):
        self.dimensions=dimensions
        self.modes=1
        mu_1 = zeros(self.dimensions)
        sigma_1 = self.getSIGMA_N(ones(self.dimensions), 0.9)
        BlobExperiment.__init__(self, n_training, n_test, [mu_1], [sigma_1], 
                            None, "e6_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,1))

class Experiment7(BlobExperiment):
    r"""
    Bimodal ND
    Density F6, F7, G [1,2].
    
    References
    ----------
    [1] Zougab, N., Adjabi, S. and Kokonendji, C.C., 2014. Bayesian estimation of adaptive bandwidth matrices in multivariate kernel density estimation. Computational Statistics & Data Analysis, 75, pp.28-38. http://www.sciencedirect.com/science/article/pii/S0167947314000322
    [2] de Lima, M.S. and Atuncar, G.S., 2011. A Bayesian method to estimate the optimal bandwidth for multivariate kernel estimator. Journal of Nonparametric Statistics, 23(1), pp.137-148. http://www.tandfonline.com/doi/pdf/10.1080/10485252.2010.485200
    """
    def __init__(self, dimensions=3, n_training=300, n_test=2000):
        self.dimensions=dimensions
        self.modes=2
        d = self.dimensions
        mu_1 = ones(self.dimensions)*4. # [4., 4., 4.]
        mu_2 = ones(self.dimensions)*4. #[1., 1., 1.]
        rho = 0.9
        c = cartesian([arange(d), arange(d)] )
        exp = abs(c[:,0]-c[:,1]).reshape((d,d))
        sigma_1 = (ones((d,d))*rho)**exp
        #sigma_1 = [[1.,    rho,rho**2],
        #           [rho,   1., rho   ],
        #           [rho**2,rho,1.    ]]
        sigma_1 = array(sigma_1)/(1-rho**(d-1))
        rho = 0.7
        sigma_2 = (ones((d,d))*rho)**exp
        #sigma_2 = [[1.,    rho,rho**2],
        #           [rho,   1., rho   ],
        #           [rho**2,rho,1.    ]]
        sigma_2 = array(sigma_2)/(1-rho**(d-1))
        BlobExperiment.__init__(self, n_training, n_test, [mu_1,mu_2], [sigma_1,sigma_2], 
                            None, "e7_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,2))

class Experiment8(BlobExperiment):
    r"""
    Bimodal 4D
    Density F8, H [1,2].
    
    References
    ----------
    [1] Zougab, N., Adjabi, S. and Kokonendji, C.C., 2014. Bayesian estimation of adaptive bandwidth matrices in multivariate kernel density estimation. Computational Statistics & Data Analysis, 75, pp.28-38. http://www.sciencedirect.com/science/article/pii/S0167947314000322      
    [2] de Lima, M.S. and Atuncar, G.S., 2011. A Bayesian method to estimate the optimal bandwidth for multivariate kernel estimator. Journal of Nonparametric Statistics, 23(1), pp.137-148. http://www.tandfonline.com/doi/pdf/10.1080/10485252.2010.485200
    """
    def __init__(self, n_training=400, n_test=2000):
        self.dimensions=4
        self.modes=2
        mu_1 = [1., 1., 1., 1.]
        mu_2 = [-1., -1., -1., -1.]
        sigma_1 = self.getSIGMA_N([1.,1.,1.,1.], 0.6)
        sigma_2 = [[1.0,0.5,0.7,0.5],
                   [0.5,1.0,0.5,0.7],
                   [0.7,0.5,1.0,0.5],
                   [0.5,0.7,0.5,1.0]]
        BlobExperiment.__init__(self, n_training, n_test, [mu_1,mu_2], [sigma_1,sigma_2], 
                            None, "e8_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,2))

class ExperimentE(EllipsoidExperiment):
    r"""
    Infmodal ND
    """
    def __init__(self, dimensions=2, n_training=200, n_test=2000):
        self.dimensions = dimensions
        self.modes=0
        radii = arange(dimensions,0,-1)
        cov=0.05
        #sigma_inner = diag(ones(dimensions)*0.2)
        EllipsoidExperiment.__init__(self, n_training, n_test, radii, cov, 
                            "eE_n(%s)_d(%s)_mode(inf)"%(n_training,self.dimensions))

class ExperimentC(BallExperiment):
    r"""
    Infmodal ND
    """
    def __init__(self, dimensions=2, n_training=200, n_test=2000):
        self.dimensions = dimensions
        mean = zeros(dimensions)
        cov = diag(arange(0.9,0.1,-0.8/self.dimensions))
        #sigma_inner = diag(ones(dimensions)*0.2)
        BallExperiment.__init__(self, n_training, n_test, mean, cov, 
                            10, "eC_n(%s)_d(%s)_mode(inf)"%(n_training,self.dimensions))

class ExperimentD(BlobExperiment):
    r"""
    Random M Modal ND
    """
    def __init__(self, modes=1, dimensions=2, n_training=200, n_test=2000):
        self.dimensions = dimensions
        self.modes = modes
        mu_s = []
        sigma_s = []
        for i in range(modes):
            sigma_i = make_spd_matrix(self.dimensions)
            #std = diag(sigma_i)**0.5
            #rho = (sigma_i/outer(std,std))
            #sigma_i =  outer((std/std.max()/self.modes), (std/std.max()/self.modes)) * rho
            sigma_s.append(sigma_i)
            low = -0.5
            high = 0.5
            mu_s.append(uniform(low=low, high=high, size=(self.dimensions)) )
        ratios = uniform(high=1./modes, size=(modes))
        ratios[:] = 1./modes #ratios.sum()
        BlobExperiment.__init__(self, n_training, n_test, mu_s, sigma_s, 
                            ratios, "eD_n(%s)_d(%s)_mode(%s)"%(n_training,self.dimensions,self.modes))

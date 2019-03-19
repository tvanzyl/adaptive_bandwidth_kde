# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:34:58 2016

@author: tvzyl
"""

import samplepoint
import mvn
import balloon
import mlloo
import visualise
import data
import partition
import cluster
import bayesian
import design

from pandas import DataFrame
import numpy as np
from numpy import mean, log, diag, eye, rot90, dot, array
from numpy.random import uniform

from sklearn import preprocessing

from scipy.stats import gaussian_kde
from scipy import matrix
#import statsmodels.api as sm

experiment1 = design.Experiment1()
dataFrame = experiment1.train
getDensity = experiment1.actualsEstimator.predict
test_set = experiment1.uniform_test

#print("silverman", mvn.getGlobalBandwidth('silverman', dataFrame))
#print("scott", mvn.getGlobalBandwidth('scott', dataFrame))
#print("cross validation maximum likelihood", mvn.getGlobalBandwidth('cv_ml', dataFrame))
#print("cross validation maximum likelihood", mvn.getCrossValidationLeastSquares(dataFrame).x)
#print("cross validation least squares", mvn.getGlobalBandwidth('cv_ls', dataFrame))
#print("rule of thumb", mvn.getGlobalBandwidth('normal_reference', dataFrame))
#print("rule of thumb", mvn.getGlobalBandwidth('silverman', dataFrame))
#print("over", mvn.getGlobalBandwidth('over', dataFrame))


x = np.array([[10,2,],
              [51,6,],
              [50,6,],], dtype=np.float64)

y = np.array([[ 1, 2,],
              [35,36,],
              [ 5, 6,],
              [23,26,],], dtype=np.float64)

VI = np.linalg.inv(np.array([[1.0,0.2],
                             [0.2,1.0]]))

import numpy.testing as npt
from mvn import mahalanobisdist
from scipy.spatial.distance import cdist

npt.assert_almost_equal(mahalanobisdist(x, y, VI), cdist(x,y,'mahalanobis', VI=VI))


#Must be squared since we want a covariance matrix
h, cov, H = mvn.getGlobalBandwidth('silverman', dataFrame)
f_ones = getDensity(dataFrame)
 
f_sil = mvn.getSamplePointDensity(dataFrame, cov, test_set)
f_sim = mvn.getSamplePointDensity(dataFrame, H, test_set)
k = gaussian_kde(dataFrame.values.T, 'silverman')
f_sci = k(test_set.T)
#l = sm.nonparametric.KDEMultivariate(data=dataFrame.values.T, var_type='c'*len(dataFrame.columns), bw='normal_reference')
#f_stm = l.pdf(test_set.T)

assert( abs(mvn.getSamplePointDensity(dataFrame, np.diag(cov), test_set) - f_sil < 1e-10).all())
assert( abs(mvn.getSamplePointDensity(dataFrame, k.covariance, test_set) - f_sci < 1e-10).all())
#assert( abs(mvn.getSamplePointDensity(dataFrame, l.bw**2, test_set) - f_stm < 1e-10).all())
assert( abs(mvn.getBalloonDensity(dataFrame.values, cov, test_set.values, True) - f_sil < 1e-10).all())
assert( abs(mvn.getBalloonDensity(dataFrame.values, H, test_set.values, True) - f_sim < 1e-10).all())

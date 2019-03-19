# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:30:39 2016

@author: tvzyl
"""

from sklearn.neighbors import KDTree, BallTree
from sklearn.model_selection import KFold
import mvn
import numpy as np
from numpy import atleast_3d, atleast_2d, outer, allclose
import cluster

def KFoldMeanLikelyHood(dataFrame, H_diag, k=3):
    nvec, ndim = dataFrame.shape
    kf = KFold(nvec, n_folds=k, shuffle=True)
    v = []
    for train, test in kf:
        v.append( np.log( mvn.getSamplePointDensity(dataFrame.iloc[train], H_diag[train], dataFrame.iloc[test]) ).mean() )
    return np.mean(v)

def getBandwidths(dataFrame, pilot, alpha, pilotfactor, optimiser, maxit, k, dist):
    kf_ll = None
    nvec, ndim = dataFrame.shape

    n, d = dataFrame.shape
    M = d-1
    if  k=='sqrt':
        k=int(n**0.5)
    elif k=='kung':
        d2 = np.max([d,2])
        k = int(M*n**(1/d2))
    elif k=='hansen':
        k = int(n**(4/(4+d)))
    if pilotfactor=='breiman':
        #knn
        if len(pilot.shape) == 2:
            newBW = np.ones( (nvec,)+pilot.shape ) #* pilot #np.diag(np.ones(ndim))
        else:
            newBW = np.ones( (nvec,)+pilot.shape ) #* pilot
        if dist is None:
            kdt = KDTree(dataFrame, leaf_size=30, metric='euclidean')
            dist, nn = kdt.query(dataFrame, k=k+1, return_distance=True)
        fx = dist[:,k]
        factor = (newBW.T*(fx/1.)**-alpha).T #newBW is the ones
    elif pilotfactor=='wu':
        if len(pilot.shape) == 2:
            newBW = np.ones( (nvec,)+pilot.shape ) * np.diag(np.ones(ndim))
            factor = pilot*cluster.getSamplePointFactor(dataFrame, 'H3')
        else:
            newBW = np.ones( (nvec,)+pilot.shape )
            factor = pilot*cluster.getSamplePointFactor(dataFrame)
    elif pilotfactor=='silverman':      
        #geometric mean    
        newBW = np.ones( (nvec,)+pilot.shape )
        fx = mvn.getLOODensity(dataFrame.values, pilot*newBW)
        sf = np.exp(np.mean(np.log(fx)))
        factor = pilot*(newBW.T*(fx/sf)**-alpha).T #newBW is the ones        
    else:
        raise NotImplementedError(pilotfactor)
    if optimiser == 'substitute':
        newBW[:] = factor
    elif optimiser == 'cv_ls':
        res = mvn.getCrossValidationLeastSquares(dataFrame, factor, 1.)
        newBW[:] = factor*res
    elif optimiser == 'cv_ls_ndim':
        if len(pilot.shape) == 2:  #covariance_class=='H3':
            #Square covariance matrix            
            res = mvn.getCrossValidationLeastSquares(dataFrame, factor, np.ones((ndim,ndim)))
        else:
            res = mvn.getCrossValidationLeastSquares(dataFrame, factor, np.ones(ndim))
        #check if this is a H3 or H2
        newBW[:] = factor*res
    elif optimiser == 'iterate':
        newBW[:] = factor
        for i in range(maxit):
            kf_ll_tmp = KFoldMeanLikelyHood(dataFrame, newBW)
            if kf_ll == None or kf_ll_tmp > kf_ll:
                kf_ll = kf_ll_tmp                
                if pilotfactor=='silverman':
                    #geometric mean
                    fx = mvn.getLOODensity(dataFrame.values, newBW)
                    sf = np.exp(np.mean(np.log(fx)))
                    factor = (newBW.T*(fx/sf)**-alpha).T #newBW is the previous factor
                    newBW[:] = factor
            else:
                print('converged in %s iterations:'%i)
                break
    else:
        raise NotImplementedError(optimiser)
    return newBW
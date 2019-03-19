# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:00:28 2016

@author: tvzyl
"""
import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import chi2

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

import mvn
import samplepoint

from math import floor, ceil

#from ellipsoid import Vk, mve
from ellipsoidpy import mve, Vk

def getDensity(samples, k, points, balloon='loftsgaarden-knn', 
               tree=None, dist=None, nn=None, dist_loo=None, nn_loo=None,
               maxjobs=None, percentile=0.6826,
               leave_out_n=0, tolerance=0.001): #0.001
    #leave_n_out=0 means dont leave any out
    #leave_n_out=1 is a loo density estimate
    if tree == None:
        kdt = KDTree(samples, leaf_size=30, metric='euclidean')
    else:
        kdt = tree
    n, d = samples.shape
    m, _ = points.shape
    M = 2
    if isinstance(k, int):
        pass
    elif k=='sqrt':
        k=int(n**0.5)
    elif k=='kung':
        d2 = np.max([d,2])
        k = M*int(n**(1/d2))
    elif k=='hansen': #Mack et.al, 'fukunaga'
        k = int(n**(4/(4+d)))
    elif k=='loo_ml':
        if dist_loo is None or nn_loo is None:
            dist_loo, nn_loo = kdt.query(samples, k=n, return_distance=True)
        k = int(n**0.5)-1
        k_old = k
        k_new = 0
        while k > 2:             
            k_new = n/((dist_loo[:, k+1] - dist_loo[:, k])/dist_loo[:, k]).sum()
            print(k, k_new)            
            k = round(int(k_new))
            if  k_old - k_new < 1.0: 
                break
            k_old = k_new
        
        print('Best k: %s'%k)
        
    elif k=='loo_cv':
        if dist_loo is None or nn_loo is None:
            dist_loo, nn_loo = kdt.query(samples, k=n, return_distance=True)
        
        estimator = mvn.GlobalKDE('rule-of-thumb', covariance='H3')
        estimator.fit(samples)
        glo_res = estimator.predict(samples)
        if balloon != 'terrel-knn':
            k = 2
            min_res = getDensity(samples, k, samples, balloon=balloon, 
                                 tree=kdt, dist=dist_loo, nn=nn_loo, 
                                 maxjobs=maxjobs, percentile=percentile, 
                                 leave_out_n=1)
            min_los = np.linalg.norm(glo_res-min_res)
#            max_n = int(n**(4/(4+d)))
            max_n = int(n**0.5)
            strike = 0 
            for loo_cv_k in range(3, max_n):
                res = getDensity(samples, loo_cv_k, samples, balloon=balloon, 
                                 tree=kdt, dist=dist_loo, nn=nn_loo, 
                                 maxjobs=maxjobs, percentile=percentile, 
                                 leave_out_n=1)
                los = np.linalg.norm(glo_res-res)
                if los < min_los:
                    min_los = los
                    k = loo_cv_k
                    strike = 0 
                elif strike < 3:
                    strike += 1
                else:
                    break
        else:
            min_k = 1
            max_k = int(n**0.5)
            k = floor((min_k+max_k)/2.0)
            min_res = getDensity(samples, k, samples, balloon=balloon, 
                                 tree=kdt, dist=dist_loo, nn=nn_loo, 
                                 maxjobs=maxjobs, percentile=percentile, 
                                 leave_out_n=1, tolerance=0.05)
            min_los = np.linalg.norm(glo_res-min_res)
            left_los = min_los
            right_los = min_los
            while True:
                #evaluate left half
                left_k = floor((min_k+k)/2.0)
                if left_k > min_k:
                    min_res = getDensity(samples, left_k, samples, balloon=balloon, 
                                     tree=kdt, dist=dist_loo, nn=nn_loo, 
                                     maxjobs=maxjobs, percentile=percentile, 
                                     leave_out_n=1, tolerance=0.05)
                    left_los = np.linalg.norm(glo_res-min_res)
                #evaluate right half
                right_k = ceil((k+max_k)/2.0)
                if right_k < max_k:
                    min_res = getDensity(samples, right_k, samples, balloon=balloon, 
                                     tree=kdt, dist=dist_loo, nn=nn_loo, 
                                     maxjobs=maxjobs, percentile=percentile, 
                                     leave_out_n=1, tolerance=0.05)
                    right_los = np.linalg.norm(glo_res-min_res)            
                #debug
                print(min_k, left_k, k, right_k, max_k)
                print(left_los, min_los, right_los)
                #pick largest decent, assume convexity
                if left_los < min_los:
                    #go left
                    max_k = k
                    k = left_k
                    min_los = left_los
                elif right_los < min_los:
                    #go right
                    min_k = k
                    k = right_k
                    min_los = right_los
                else:
                    min_k = left_k
                    max_k = right_k
                
                if min_k+1 >= k >= max_k-1:
                    break
                
        print('Best k: %s'%k)
    
    if dist is None or nn is None:        
        dist, nn = kdt.query(points, k=k, return_distance=True)
    
    if balloon=='loftsgaarden-knn':   #Fisrt proposed by Mack et.al "Multivariate k-Nearest Neighbor Density Estimates"
        r = dist[:,k-1+leave_out_n]
        return k/(n*Vk(r,d))
    elif balloon=='kung-knn':
        r = dist[:,k-1+leave_out_n]
        return (k-1.)/(n*Vk(r,d))
    elif balloon=='biau-knn':  #Biau et al. (2011)
        return (k*(1.0+k))/(2.0*n*np.sum(Vk(dist[:,:k-1+leave_out_n],d), axis=1))
    elif balloon=='loftsgaarden-kernel-knn':
        r = dist[:,k-1+leave_out_n]
        # \lambda_i = \frac{r^2}{c} 
        # where c is your confidence interval
        lambde = r**2/chi2.ppf(percentile, d)
        hk = np.repeat( lambde, d ).reshape((m,d))        
        return mvn.getBalloonDensity(samples.values, hk, points.values, True)
#    elif balloon=='terrel-pilot-knn': #Terrell and Scott (1992)
#        # Page 1258, 
#        # Here Terrell et.al. suggest the optimal h_k 
#        # We use the rule of thumb pilot for f(y)
    elif balloon== 'terrel-knn':
        # Page 1258
        # Here Terrell et.al. suggest using the minimum enclosing ellipsoid
#        _, nn_ter = kdt.query(points, k=k, return_distance=True)
        hk = mve(samples.values, k, points.values, nn, m, d, leave_out_n, tolerance)
        return (k/(n*Vk(1,d)*(hk.prod(axis=1))))
    elif balloon== 'biau-ellipse-knn':
        # Page 1258
        # Here Terrell et.al. suggest using the minimum enclosing ellipsoid
#        _, nn_ter = kdt.query(points, k=k, return_distance=True)
        hk = mve(samples.values, k, points.values, nn, m, d, leave_out_n)
        return (k*(1.0+k))/(2.0*n*np.sum(Vk(hk,d), axis=1))
    elif balloon=='mittal': #Supposed Hybrid        
        estimator = samplepoint.SamplePointKDE(covariance='H2')
        estimator.fit(samples)
        hki = estimator.H_
        r = dist[:,k-1+leave_out_n]
        lambde = r**2/chi2.ppf(0.80, d)
        hk = np.repeat( lambde, d ).reshape((m,d))
        return mvn.getSamplePointDensity(samples, hki, points)*mvn.getBalloonDensity(samples.values, hk, points.values, True)
        #return np.array([mvn.getSamplePointDensity(samples, hk[i]+hki, points[i:i+1]) for i in range(m)])
    elif balloon=='hall-pilot-knn':
        #Get a global bandwidth for the 
        k = np.max([d+1,k])
        _, nn_hall = kdt.query(points, k=k, return_distance=True)
        hk = np.array([mvn.getGlobalBandwidth('rule-of-thumb', samples.iloc[nn_hall[i]])[2] for i in range(m)])
        return mvn.getBalloonDensity(samples.values, hk, points.values, True)

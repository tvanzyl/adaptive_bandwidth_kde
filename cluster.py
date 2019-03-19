# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:10:29 2016

@author: tvzyl
"""

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

import mvn
from sklearn.base import BaseEstimator

def getSamplePointFactor(dataFrame, covariance_class='H2'):
    r"""
    
    References
    ----------
    [1] Wu et al. (2007)
        A variable bandwidth selector in multivariate kernel density estimation 
        http://www.stat.ncku.edu.tw/faculty_private/tjwu/varbandnew.pdf
    """
    _, d = dataFrame.shape
    b = []
    for dim in range(d):
        Z = hierarchy.average(dataFrame[[dim]])
        #hierarchy.dendrogram(Z)
        nsd = {}
        n=Z.shape[0]+1
        ns = np.zeros((n,3))
        ns[:,0] = np.arange(0,n)    
        def update(t, l):
            if t>=n:
                update(int(Z[t-n,0]), l)
                update(int(Z[t-n,1]), l)
            else: 
                ns[t,1] += 1
                ns[t,2] += l
                if t in nsd:
                    nsd[t].append(l)
                else:
                    nsd[t] = [l]
        
        l = n
        for i in range(n-2,-1,-1):
            #print (Z[i])
            if Z[i,0] < n and Z[i,1] < n: 
                #update(Z[i,0], 0)
                #update(Z[i,1], 0)
                pass
            else:
                update(int(Z[i,0]), l)
                update(int(Z[i,1]), l)
                l-=1
        b.append(ns[:,2]/ns[:,1]-l)
    B = np.array(b).T
    B_mean = B.mean(axis=0)
    if covariance_class=='H3':
        return  np.asarray([np.outer(f, f) for f in B/B_mean])
    elif covariance_class=='H2':
        return (B/B_mean)**2.0
    else:
        raise NotImplementedError("wu factor doesn't support H1")
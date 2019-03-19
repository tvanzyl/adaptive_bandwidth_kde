from __future__ import division

import numpy as np
#from scipy import linalg
#from math import factorial

from numba import jit, float64, int64, prange, generated_jit

from multiprocessing import Pool, cpu_count

factorial = np.array([
1, 1, 2, 6, 24, 120, 720, 5040, 40320,
362880, 3628800, 39916800, 479001600,
6227020800, 87178291200, 1307674368000,
20922789888000, 355687428096000, 6402373705728000,
121645100408832000, 2432902008176640000], dtype='int64')


#@jit(nopython=True)
def Vk(R, d):    
    if   d==1:
        return (2.)*R
    elif d==2:
        return np.pi*R**2.
    elif d==3:
        return (2.**2)*np.pi**1./3.*R**3.
    elif d==4:
        return np.pi**2/2*R**4.
    elif d==5:
        return (2.**3)*np.pi**2./15.*R**5.
    elif d==6:
        return np.pi**3./6.*R**6
    elif d==7:
        return (2.**4)*np.pi**3./105.*R**7.
    elif d==8:
        return np.pi**4./24.*R**8
    elif d==9:
        return (2.**5)*np.pi**4./945.*R**9.
    elif d==10:
        return np.pi**5./120.*R**10.
    elif d in [11,13,15,17,19]:
        k = int((d-1)/2)
        return 2*factorial[k]*(4*np.pi)**k/factorial[d]*R**d
    elif d in [12,14,16,18,20]:
        k = int(d/2)
        return np.pi**k/factorial[k]*R**d
    elif d <= 50:
        #Use recurance relation
        return 2.*np.pi*R**2./d*Vk(R, d-2)
    else:
        #Use the high dimensional approximation
        return (d*np.pi)**-0.5*(2.*np.pi*np.e/d)**(d/2.)*R**d

def Sk(R, d):
    if d==0:
        R[:] = 2.
        return R
    else:
        return 2.*np.pi*Vk(np.ones(1), d)*R

# -*- coding: utf-8 -*-
@jit(float64[:](float64[:,:],float64), nopython=True, nogil=True)
def getMinVolEllipse(P, tolerance):
    """ Find the minimum volume ellipsoid which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    (N, dim) = P.shape    
    d = float(dim)

    # Q will be our working array
    # Q = np.vstack((np.copy(P.T), np.ones(N))) 
    Q = np.ones((dim+1,N), dtype=np.float64)
    Q[:dim] = P.T
    
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)
    
    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        # M the diagonal vector of an NxN matrix
        M = np.diag(np.dot(QT, np.dot(np.linalg.inv(V), Q)))
        j = M.argmax()
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        err = np.linalg.norm(new_u - u)
        if err <= tolerance:
            break
        new_u[j] += step_size
        u = new_u

    PT = P.T
    
    # center of the ellipse 
    center = np.atleast_2d(np.dot(PT, u))
    
    # the A matrix for the ellipse
    A = np.linalg.inv(
                   np.dot(PT, np.dot(np.diag(u), P)) - 
                   np.outer(center, center)
#                   np.array([[a * b for b in center] for a in center])
                   ) / d
                   
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0/np.sqrt(s)    
    radii *= 1. + tolerance
    
    return radii

# -*- coding: utf-8 -*-
@jit(float64[:](float64[:,:],float64), nopython=True, nogil=True)
def getMinVolEllipse_pinv(P, tolerance):
    """ Find the minimum volume ellipsoid which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    (N, dim) = P.shape    
    d = float(dim)

    # Q will be our working array
#    Q = np.vstack((np.copy(P.T), np.ones(N))) 
    Q = np.ones((dim+1,N), dtype=np.float64)
    Q[:dim] = P.T
    
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)
    
    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        # M the diagonal vector of an NxN matrix
        M = np.diag(np.dot(QT, np.dot(np.linalg.pinv(V), Q)))    
        j = M.argmax()
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        err = np.linalg.norm(new_u - u)
        if err <= tolerance:
            break
        new_u[j] += step_size
        u = new_u
    
    PT = P.T
    
    # center of the ellipse 
    center = np.atleast_2d(np.dot(PT, u))
    
    # the A matrix for the ellipse
    A = np.linalg.pinv(
                   np.dot(PT, np.dot(np.diag(u), P)) - 
                   np.outer(center, center)
#                   np.array([[a * b for b in center] for a in center])
                   ) / d
    
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0/np.sqrt(s)    
    radii *= 1. + tolerance
    return radii

#@jit(float64[:](float64[:,:],float64[:]),
#     nopython=True, nogil=True)
def mve_inv(samples, point, tolerance):
#    refKDataFrame = np.r_[kDataFrame, point[np.newaxis, :]] 
    refsamples = np.vstack((samples, 2.*point - samples))
    radii = getMinVolEllipse(refsamples, tolerance)
    return radii

#@jit(float64[:](float64[:,:],float64[:]),
#     nopython=True, nogil=True)
def mve_pinv(samples, point, tolerance):
#    refKDataFrame = np.r_[kDataFrame, point[np.newaxis, :]] 
    refsamples = np.vstack((samples, 2.*point - samples))
    radii = getMinVolEllipse_pinv(refsamples, tolerance)
    return radii

#@jit(float64[:,:](float64[:,:],int64,float64[:,:],int64[:,:],int64,int64,int64),
#     nopython=True, nogil=True, parallel=True)
def mve(samples, k, points, nn, m, d, leave_out_n, tolerance):
#    hk  = np.empty((m, d), dtype=np.float64)
#    if k>d:
#        for i in prange(m):
#            hk[i,:] = mve_inv(samples[nn[i,leave_out_n:k-1+leave_out_n]], points[i])
#    else:
#        for i in prange(m):
#            hk[i,:] = mve_pinv(samples[nn[i,leave_out_n:k-1+leave_out_n]], points[i])
    mve_f = mve_inv if k>d else mve_pinv
    Pooler = Pool
    with Pooler() as pool:
        hk = np.array(pool.starmap(mve_f,
                                   ((samples[nn[i,leave_out_n:k-1+leave_out_n]], points[i], tolerance) for i in range(m)),
                                   chunksize=int(m/cpu_count())))
    return hk

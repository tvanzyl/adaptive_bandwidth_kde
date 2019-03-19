#!/usr/bin/python
# cython: profile=True
from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

#from scipy import linalg
#import numpy.linalg
#cimport numpy.linalg

from random import random
from math import factorial

DTYPE = np.float
ctypedef np.float_t DTYPE_t

#https://en.wikipedia.org/wiki/Volume_of_an_n-ball
#Volume of a n dimensional hypeshere in L2
cpdef Vk(np.ndarray R, int n):
    assert R.dtype == DTYPE
    if n==1:
        return (2.)*R
    elif n==2:
        return np.pi*R**2.
    elif n==3:
        return (2.**2)*np.pi**1./3.*R**3.
    elif n==4:
        return np.pi**2/2*R**4.
    elif n==5:
        return (2.**3)*np.pi**2./15.*R**5.
    elif n==6:
        return np.pi**3./6.*R**6
    elif n==7:
        return (2.**4)*np.pi**3./105.*R**7.
    elif n==8:
        return np.pi**4./24.*R**8
    elif n==9:
        return (2.**5)*np.pi**4./945.*R**9.
    elif n==10:
        return np.pi**5./120.*R**10.
    elif n in [11,13,15]:
        k = (n-1.)/2.
        return 2*factorial(k)*(4*np.pi)**k/factorial(n)*R**n
    elif n in [12,14,16]:
        k = n/2.
        return np.pi**k/factorial(k)*R**n
    elif n <= 20:
        #Use recurance relation
        return 2.*np.pi*R**2./n*Vk(R, n-2)
    else:
        #Use the high dimensional approximation
        return (n*np.pi)**-0.5*(2.*np.pi*np.e/n)**(n/2.)*R**n

cpdef Sk(np.ndarray R, int n):
    assert R.dtype == DTYPE
    if n==0:
        R[:] = 2.
        return R
    else:
        return 2.*np.pi*Vk(np.ones(1, dtype=DTYPE), n)*R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getMinVolEllipse(np.ndarray[DTYPE_t, ndim=2] P not None, float tolerance):
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
    cdef float err = 1.0 + tolerance
    
    cdef int N    
    cdef int dim
        
    (N, dim) = np.shape(P)
    cdef float d = float(dim)

    cdef np.ndarray[DTYPE_t, ndim=2] PT = P.T
    
    # Q will be our working array
    cdef np.ndarray[DTYPE_t, ndim=2] Q = np.vstack([np.copy(PT), np.ones(N, dtype=DTYPE)])
    cdef np.ndarray[DTYPE_t, ndim=2] QT = Q.T
    
    # initializations
    cdef np.ndarray[DTYPE_t, ndim=1] u = (1.0 / N) * np.ones(N, dtype=DTYPE)    
    cdef np.ndarray[DTYPE_t, ndim=1] new_u = np.ndarray(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] V = np.ndarray((dim+1, dim+1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] M = np.ndarray(N, dtype=DTYPE)
    
    cdef DTYPE_t maximum
    cdef DTYPE_t step_size
    cdef int j
    # Khachiyan Algorithm
    while err > tolerance:
        #V = np.dot(Q, np.dot(np.diag(u), QT))
        V[:] = Q.dot(u[:,np.newaxis]*QT)
        M[:] = QT.dot(np.linalg.inv(V).dot(Q)).diagonal()    # M the diagonal vector of an NxN matrix
        j = M.argmax()
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u[:] = (1.0 - step_size) * u
        new_u[j] += step_size
#        err = np.linalg.norm(new_u - u)
        err = ((new_u - u)**2.).sum()**0.5
        u[:] = new_u

    # center of the ellipse 
    cdef np.ndarray[DTYPE_t, ndim=1] center = PT.dot(u)
    
    # the A matrix for the ellipse
    A = np.linalg.inv(
                   PT.dot(u[:,np.newaxis]*P) - 
#                   np.dot(P.T, np.dot(np.diag(u), P)) - 
#                   np.array([[a * b for b in center] for a in center], dtype=DTYPE)
                   np.outer(center,center)
                   ) / d
                   
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    
    cdef np.ndarray[DTYPE_t, ndim=1] radii = 1.0/(s**0.5)    
    return (center, radii, rotation)




def mve(np.ndarray[DTYPE_t, ndim=2] kDataFrame not None, np.ndarray[DTYPE_t, ndim=1] point not None):
    cdef np.ndarray[DTYPE_t, ndim=2] refKDataFrame = np.r_[kDataFrame, (2.*point - kDataFrame)]
    (_, radii, _) = getMinVolEllipse(refKDataFrame, 0.001)
    return radii


#data = pd.DataFrame(sklearn.datasets.make_blobs(centers=1)[0])
#(center, radii, rotation) = getMinVolEllipse(data.values)
#
#ax = data.plot.scatter(x=0,y=1)
#dimensions_ = 2
#
#var_ = 0.05
#n_samples = 200
#x = np.random.normal(size=(n_samples,dimensions_))
#r = la.norm(x, axis=1)
##pd.DataFrame(1./r[:,np.newaxis]*x).plot.scatter(0,1,ax=ax,color='orange')
#e = np.dot(np.dot(1./r[:,np.newaxis]*x, np.diag(radii)), rotation)+center
#pd.DataFrame(e).plot.scatter(0,1,ax=ax,color='red')

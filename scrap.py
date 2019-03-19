#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:05:25 2019

@author: tvzyl
"""
import numpy as np

###
# balloon terrel-knn H3 loo_cv
# Best k: 47
# time elapsed: 772.42370 ISE: 0.026593743        
# balloon biau-knn H1 loo_cv
# Best k: 23
# time elapsed: 2.86176 ISE: 0.032016206
# balloon kung-knn H1 loo_cv
# Best k: 15
# time elapsed: 2.79583 ISE: 0.039454231
# balloon loftsgaarden-knn H1 loo_cv
# Best k: 17
# time elapsed: 2.69768 ISE: 0.037240348
# balloon loftsgaarden-kernel-knn H1 loo_cv
# Best k: 2
# time elapsed: 3.03139 ISE: 0.101104007
### 

def getMinVolEllipseAxis(S):
    nrow, ndim = S.shape
    alphaP = np.argmax(S, axis=0)
    alphaN = np.argmin(S, axis=0)
    alpha = np.unique( np.hstack((alphaN, alphaP)) )
    alpha_norm = len(alpha)
    
    X = np.zeros((nrow), dtype=bool)
    X[alpha] = True
    
    sigma = np.zeros((nrow,1))
    sigma[alpha] = 1.0/alpha_norm
    
    epsln = 0.001
    
    k = 0
    while True:
        k += 1 
        u = (sigma*(S**2)).sum(axis=0)
        v = (sigma*S).sum(axis=0)
        d = 1.0/(ndim*(u-v**2))
                
        E = (((S-v)**2)*d).sum(axis=1)
        i = np.argmax( E )
        e = E[i] - 1.0
        if  e <= (1+epsln)**(2/ndim) - 1:
            break 
        else:
             X[i] = True
             beta = e / ((ndim+1)*(1+e))
             sigma = (1-beta)*sigma + beta
    print(k)
    Vk(1, 2)*(1/d**0.5).prod()

arr = np.array([[0,0,0],
                [0,0,2],
                [0,1,0],
                [0,1,2],
                [1,0,0],
                [1,0,2],
                [1,1,0],
                [1,1,2]])-1.0

arr = np.array([[ 0, 0],
                [ 0, 2],
                [ 1, 0],
                [ 1, 2]])
S = arr

center = np.array([1.0,1.0])

#getMinVolEllipse(arr*1.0, 0.001).prod()*Vk(1,2)

getMinVolEllipse(np.r_[arr, 2*center-arr], 0.001)

x = np.array([[1,2,],
              [5,6,],
              [5,6,],], dtype=np.float64)

y = np.array([[ 1, 2,],
              [35,36,],
              [ 5, 6,],], dtype=np.float64)

VI = np.linalg.inv(np.array([[1., 0.2],
                             [0.2,1.]]))



class A:
    def __call__(self, x):
        print(X)
        
a = A()
a(1)

@jit(nb.int32(nb.float64[:,:]), nopython=True)
def A2(a):
    return 2


@jit(nb.int32(nb.float64[:,:,:]), nopython=True)
def A3(a):
    return 3
    
@generated_jit(nopython=True)
def A(a):
    if a == types.Array(types.float64, 2, 'C'):
        return A2
    elif a == types.Array(types.float64, 3, 'C'):
        return A3
    else:
        raise ValueError()



import sympy as sym
sym.init_printing()

i, j, n, V_d, d, sigma, k = sym.symbols('i j n V_d d sigma k')
#x = sym.IndexedBase('x', shape=(m, d))
x = sym.IndexedBase('x', shape=(n,))
K_nn = sym.Function('K_nn')
#L_2 = sym.Sum( (K_nn(k,x[i, j]) - x[i, j])**2, (j, 1, d) )**0.5 
L_2 = sym.Function('L_2')(x[i], x, k)


f = (k**d/(n*V_d*L_2**d))

l = 1/n*sym.Sum(sym.ln(f),(i, 1, n) ) 

delta_l = sym.simplify( sym.diff(l, k) )
delta_l = sym.simplify( delta_l*n/d)
delta_l







































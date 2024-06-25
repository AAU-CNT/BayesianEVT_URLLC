# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 09:27:24 2021

@author: Tobias Kallehauge
"""

import numpy as np
from scipy.special import betainc
import warnings
from functools import lru_cache

@lru_cache(maxsize = 100000)
def _get_w(n,j,k):
    
    p = n - k + 1
    a = (j-1)/n
    b = j/n
    I = betainc(k,p,b) - betainc(k,p,a)

    return(I)
    

def _get_m(X, p, B = 100, S = 10, L = 20):
    """
    Determine m using the emperical approach from [1]
    """
    
    n = X.size
    
    ms = (n//2)*np.arange(1,S + 1)//S # can be chosen differently this is about 
    # [i, n/2 - i] for a small integer i
    ls = np.zeros(S)
    for s in range(S):
        # L numbers between 2 and ms
        L = ms[s]
        l_rng = np.unique(np.linspace(2,ms[s], L).astype('int')) 
        L = l_rng.size 
        
        # compute weights in seperate loop
        w = np.zeros((L,ms[s]))
        for i, l in enumerate(l_rng):
            k = int(l*p) + 1
            w[i] = np.array([_get_w(ms[s], l, j, k) \
                          for j in range(1, ms[s] + 1)])
        
            # plt.plot(w[i]); plt.show()
        _ , var_s = quantile_var_boostrap(X, p, method = 'm of n', m = ms[s])
        var_s = n/ms[s] # scale to ms samples only
        var_bl = np.zeros((B,L))
        r = int(ms[s]*p) + 1
        for b in range(B):
            X_sb = np.random.choice(X,ms[s])    
            X_sb_ord = np.sort(X_sb) # order statistics
            X_sb_r = X_sb_ord[r - 1] # subtract 1 since counting from zero
            for i, l in enumerate(l_rng):
                # calculate variance of bootstrap sample
                var_bl[b,i] = (l/ms[s])*((X_sb_r - X_sb_ord)**2*w[i]).sum()
        
        # get the choise of l that minimizes the difference between var_bl and 
        # var_s, denoted l_s
        err = ((var_bl - var_s)**2).mean(axis = 0)
        ls[s] = l_rng[err.argmin()]
    

    # calculate m
    M1 = np.log(ms).sum()
    M2 = (np.log(ms)**2).sum()
    L1 = np.log(ls).sum()
    K = (np.log(ms)*np.log(ls)).sum()
    D = S*M2 - M1**2
    c = np.exp((M2*L1 - M1*K)/D)
    gamma = (S*K - M1*L1)/D
    m = int(c*n**gamma)
    
    m = max(2,min(m,n)) # within [2,n]
    
    return m

def quantile_var_boostrap(X,p, method = 'n of n', **kwargs):
    """
    Estimate the variance for quantile estimation for the samle X with n 
    observations using using either n out of n or m of n boostrap method [1].
    m of n is generally a better estimator but is more complex due to the 
    procedure of finding m. 
    
    References
    ----------
    [1] Cheung, K.Y., Lee, S.M.S. Variance estimation for sample quantiles 
    using the m out of n bootstrap. Ann Inst Stat Math 57, 279–290 (2005).
    https://doi.org/10.1007/BF02507026
    
    Parameters
    ----------
    X : np.ndarray
        Scalar data with n observations. 
    p : float
        p-value for the quantile beeing estimated.   
    method : int
        Method to use. Options are "n of n" and "m of n"
    
    Returns
    -------
    X_r : float
        
    var : float
        Estimate of variance for quantile esimtation
    """
    
    if method == 'm of n':
        warnings.warn('m of n method is not stable (cannot find m)')
    
    
    n = X.size
    
    X_ord = np.sort(X) # order statistics
    r = int(np.ceil(n*p)) # the order statistic number
    X_r = X_ord[r - 1] # subtract 1 since counting from zero
    
    
    if method == 'm of n':
        if kwargs.get('m') is None: # get m if not computed already
            m = _get_m(X,p,**kwargs)
        else:
            m = kwargs.get('m')
        
        k = int(m*p) + 1
        w = np.array([_get_w(n,m, j, k) for j in range(1,n + 1)])   
    
    # get w
    elif method == 'n of n':
        if 'w' in kwargs:
            w = kwargs['w']
        else:    
            w = np.array([_get_w(n, j, r) for j in range(1,n + 1)])
            
    # compute the variance estimate
    var = ((X_ord - X_r)**2*w).sum()
    
    if method == 'm of n':
        var *= m/n
    
    return(X_r,var)


def quantile_est(X, eps):
    """
    Quantile estimate for X based on inverse CDF
    """
    
    assert X.ndim == 1, 'Only 1 dimensional arrays allowed'

    X_ord = np.sort(X)
    r = int(np.ceil(len(X)*eps))

    return(X_ord[r - 1]) # subtract 1 since counting from 0

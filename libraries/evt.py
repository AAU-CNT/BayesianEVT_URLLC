# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:27:16 2023

@author: Tobias Kallehauge
"""

import numpy as np
from scipy.stats import genpareto

def mean_residual_life(X, n = 1000, est_std = True, u_min = None, u_max = None, 
                       log_u = False):
    """
    Produce mean residual life points (u, E[u - X| X < u])
    
    u is between 0 and u_max

    Parameters
    ----------
    X : np.ndarray
        Observed values
    n : int, optional
        Number of thresholds u. The default is 1000.
    est_std : bool, optional
        If the standard deviation for each threshold should also be estimated.
    log_u : bool, optional
        If True, u is sampled logarithmically rather than linear. 
        The defalut is False

    Returns
    -------
    mrl : np.ndarray
        matrix, with first column the values of u and second column the mean
        residual. If est_std is True, a third row es returned with the standard
        deviation
    """
    
    if u_max is None:
        u_max = max(X)
    if u_min is None:
        u_min = min(X)*0.99
    
    if est_std:
        n_return = 3
    else:
        n_return = 2
    
    
    
    mrl = np.zeros((n,n_return))
    
    if log_u: 
        U = 10**(np.linspace(np.log10(u_min), np.log10(u_max),n))
    else:
        U = np.linspace(u_min, u_max, n)
    
    
    mrl[:,0] = U
    
    for i,u in enumerate(U):
        
        Y = u - X[X < u] 
        if Y.size > 0: 
            mrl[i,1] = np.mean(Y)
        
            if est_std:
                mrl[i,2] = np.sqrt(np.var(Y)/Y.size)
        else:
            mrl[i,1] = np.nan
            if est_std:
                mrl[i,2] = np.nan
        
    return(mrl)

def parameter_stability(X, n = 100, est_std = False,u_min = None, u_max = None,
                        log_u = False):
    """
    

    Parameters
    ----------
    X : np.ndarray
        Observed values
    n : int, optional
        Number of thresholds u. The default is 100.
    est_std : bool, optional
        If the standard deviation for each threshold should also be estimated.
        The default is False (not yet implemented)
    u_max : float
        Minimum tested value of u. If None, min(X) is used. 
        The default is None.
    u_max : float, optional
        Maximum tested value of u. If None, max(X) is used. 
        The default is None.
    log_u : bool, optional
        If True, u is sampled logarithmically rather than linear. 
        The defalut is False
    
    
    Returns
    -------
    u : np.ndarray
        Tested threshold u
    xi : np.ndarray
        Fitted values of xi for each level of u
    sigma_ast : np.ndarray
        Fitted values of sigma* for each level of u

    """
    
    if u_max is None:
        u_max = max(X)
    if u_min is None:
        u_min = min(X)
    
    
    
    if est_std:
        raise(NotImplementedError())

    xi = np.zeros(n)
    sigma_ast = np.zeros(n)
    
    if log_u: 
        U = 10**(np.linspace(np.log10(u_min), np.log10(u_max),n))
    else:
        U = np.linspace(u_min, u_max, n)
    
    
    for i, u in enumerate(U):
        Y = u - X[X < u] 

        if Y.size > 0: 
        
            xi_fit, _, scale_fit = genpareto.fit(Y, floc = 0) # shape, location and scale
            
            xi[i] = xi_fit
            sigma_ast[i] = scale_fit - xi_fit*u
        else:
            xi[i] = np.nan
            sigma_ast[i] = np.nan
        
    
    return(U,xi,sigma_ast)

def detect_bend(u, mrl, n = 5, tolerence = 0.2, idx_min = 10):
    """
    Detect bend in mean residual life (MRL) curve for exeedences below a
    threshold u.
    
    Theoretically, the MRL curve should be linear up till a point. This 
    function finds the first bend in the curve, which should correspond to the
    point where the curve stops beeing linear. The bend is detected using the
    second derivative of the curve looking for large negative values in the 
    derivative which corresponds to local minima. 
    Note that, in practice, it does not allways work. 

    Parameters
    ----------
    u : np.ndarray
        Threshold values
    mrl : np.ndarray
        Mean residual life values.
    n : int, optional
        How many local minima among candidate bends are evaluated.
        The default is 5.
    percent : float, optional
        There are usually several local minima. The algorithm first finds the
        bend with the smallest second derivative. Then it choses the 
        largest threshold which is 1 + percent close to the bend.        
        The default is 0.2.
    idx_min : int, optional
        The minimum index of u that are allowed to be a threshhold. 
        The default is 10.

    Returns
    -------
    u_thresh : float
        Detected thresh

    """
    
    
    du = u[1] - u[0]
    Diff = (mrl[2:] + mrl[:-2] - 2*mrl[1:-1])/du**2
    mrl = mrl[1:-1]
    u = u[1:-1]

    Diff_thin = Diff.copy()
    local_minima = np.zeros(n, dtype = 'int')
    for i in range(n):
        idx_min = np.nanargmin(Diff_thin)
        local_minima[i] = idx_min
        # remove points on the left
        j = 1
        if idx_min > 0:
            while Diff_thin[idx_min - j] > Diff_thin[idx_min - j + 1] and idx_min - j > 0:
                j += 1
            Diff_thin[idx_min - j + 1: idx_min] = np.nan
        
        # remove points on the right
        if idx_min < len(Diff) - 1:
            j = 1
            while Diff_thin[idx_min + j] > Diff_thin[idx_min + j - 1] and idx_min + j < len(Diff) - 1 :
                j += 1
            Diff_thin[idx_min + 1: idx_min + j] = np.nan
        
        #remove local minimum 
        Diff_thin[idx_min] = np.nan
    
    local_minima = local_minima[local_minima >= idx_min]
    
    # find smalles within percent of smallest minima
    thresh = Diff[local_minima[0]]*(1 - tolerence)
    bend_idx = local_minima[thresh >= Diff[local_minima]].max()

    return(u[bend_idx])
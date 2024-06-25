# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:05:44 2020

@author: Tobias Kallehauge
"""

import warnings

def newton(f_df,r0,err = 1e-8,df_min = 1e-8, tol = 1e-8,MAX_ITER = 100, 
           delta = 1, min_val = 0,
           verbose = False,**kwargs): 
    """
    Performs newtons method for finding roots of functions. 

    Parameters
    ----------
    f_df : function
        Function that returns function values to find roots for and their
        derivatives.  
    r0 : float
        Initial guess.
    tol : float, optional
        Tolerence, stop when tolerence is met. The default is 1e-10.
    MAX_ITER : int, optional
        Allways stop after MAX_ITER itereations. The default is 100.
    delta : float, optional
        Step size - sometimes smaller stepsize is needed. The default is 1.
    min_val : float, optional
        Minimum value to allow as input. Default is 0 
    verbose: bool, optional
        If verbose is True, prints out which convergence parameter i met. 
    
    Returns
    -------
    r : float
        Approximate root of function. 
    """
    
    r = r0
    
    for i in range(MAX_ITER):
        
        # evaluate functions 
        y, dy = f_df(r0)
        
        if verbose: 
            print(f"r: {r:.2f}, err: {y:.2e}")
        
        # stop of derivative is too small 
        if abs(dy) <= df_min:
            if verbose:
                print("Derivative")
            break
            
        
        # update via newtons step
        r = r0 - delta*y/dy
        
        # try and catch values less than min_val 
        if r < min_val: 
            for i in range(10):
                delta = delta/2
                r = r0 - delta*y/dy
                if r > min_val:
                    break
                
            # stop if still below minimum value 
            if r < min_val:
                r = min_val
                if verbose:
                    print('Below minimum value')
                break 
        
        # stop if below tolerence for root precission
        if abs(r0 - r) < tol:
            if verbose:
                print("Tolerence")
            break
        
        
        # stop if f error is small enough (close to zero)
        if abs(y) <= err:
            if verbose:
                print("Error")
            break

        r0 = r 
    
    if i == MAX_ITER - 1 and verbose:
        warnings.warn("Maximum number of iterations met.")
        
    return(r)
 
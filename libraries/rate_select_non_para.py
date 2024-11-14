# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 08:35:34 2022

@author: Tobias Kallehauge
"""

import numpy as np
from quantile_variance_bootstrap import quantile_var_boostrap, quantile_est
from scipy.special import betainc
from scipy.stats import norm

def rate_select_non_para(eps, X = None, model = 'capacity',  method = 'freq',
                         target = 'pcr', **kwargs):
    """
    Select communication rate conservatively based on observed values of 
    recieved signal strength (RSS) which is X. Various options are available. 

    Parameters
    ----------
    eps : float
        Target outage probability
    X : np.ndarray, optional
        Observed values of X which is either the SNR or the instantanious 
        capacity C =log2(1 + SNR) (accoridng to the model parameter). 
        If None, prior information must be supplied. 
        X should be in LINEAR domain - i.e. Watt or bits/sec/Hertz, but the 
        priors are for the logarichmic domain. 
        The default is None. 
    model : str, optional
        Which variable X is, either 'SNR' for SNR or 'capacity' for C. 
    method : str, optional
        How to select the rate. Options are:
            freq: Frequentist approach selecting the rate only based on the 
            available samples and not prior information. 
            bay : Bayesian approach where prior information is used. 
        The default is freq
    Target: str, optional
        Which criteria to select the rate based on. Options are:
            mode : Select the unbiased consisten estimator, for example the MAP
            estimate for bayesian or emperical quantile. 
            pcr : Select rate based on probabily correct reliability (pcr) with
            given confidence delta
        The default is pcr. 
        
    Optional keyword arguments
    --------------------------
    prior : dict
        Dictionary with prior information.     
    delta : float
        Confidence parameter when target is pcr. 
    
    Returns
    -------
    R : float
        Communication rate in bits/sec/Hz
    """
    
    if X is None:
        assert 'prior' in kwargs, \
        'Supply prior information when X is not observed'
    else:
        n = X.size
    if target == 'pcr':
        assert 'delta' in kwargs, \
            'Supply confidence parameter delta for prc rate selection'
        
    
    
    if method == 'freq':
        if len(X) == 0:
            q = 0
        elif target == 'mode':
            q = quantile_est(X,eps)
        elif target == 'pcr':
            l, _ = find_l(eps, kwargs['delta'], n)
            if l == 0:
                q = 0
            else:
                q = np.sort(X)[l - 1] # subtract 1 since counting from zero
                
    if method == 'bay': # prior information
        prior = kwargs['prior']
        mean_prior = prior['q_mean']
        var_prior = prior['q_var']
        
        if X is None or len(X) == 0:
            mean_post = mean_prior
            var_post = var_prior
            
        else:
            if 'pdf_eps' in prior and 'post_factor' in kwargs: # combine prior and likelihood for prior
                # get statistics based on the sample itself
                mean_sample, var_sample = quantile_var_boostrap(np.log(X),eps,**kwargs)
                var_sample_prior = eps*(1-eps)/(n*prior['pdf_eps']**2)
                
                # for slower congence to the sample variance, divide by a smaller number
                K = np.log10(n)/kwargs['post_factor']
                
                # update var_sample with prior information
                var_sample = var_sample_prior/(K+1) + var_sample*K/(K+1)
            
            elif 'pdf_eps' in prior: # only use prior
                mean_sample = quantile_est(np.log(X), eps)
                var_sample = eps*(1-eps)/(n*prior['pdf_eps']**2)
            else:
                mean_sample, var_sample = quantile_var_boostrap(np.log(X),eps,**kwargs)
            
                
            mean_post, var_post = gauss_posterior(mean_prior, var_prior,
                                                  mean_sample, var_sample)
        
        if target == 'mode':
            q_ln = mean_post
        elif target == 'pcr':
            q_ln = norm.ppf(kwargs['delta'], 
                            loc = mean_post,
                            scale = np.sqrt(var_post))
            
        q = np.exp(q_ln)

    
    if model == 'SNR':
        R = np.log2(1 + q)
    else: # model == 'capacity'
        R = q
        
    return(R)

def find_l(epsilon, delta, n):    
    # PCR
    search_val = 0
    l_pcr = 0
    while True:
        search_val = search_val + 1
        val = 1 - betainc(search_val, n + 1 - search_val, epsilon)
        if val < delta:
            l_pcr = search_val
        else:
            break
    selected_l_n = l_pcr
    # update pcr value
    if selected_l_n == 0:
        pcr = 0
    else:
        pcr = val = 1 - betainc(selected_l_n, n + 1 - selected_l_n, epsilon) 
    return selected_l_n, pcr

def gauss_posterior(mean_prior, var_prior, mean_sample, var_sample):
    """
    Get the posterior predictive distribution for random variable X
    when X and the prior is known and Gaussian. 

    Parameters
    ----------
    mean_prior : float
        Prior mean.
    var_prior : float
        Prior variance.
    mean_sample : float
        Likelihood mean (from sample)
    var_sample: float
        Likelihood variance (from sample)

    Returns
    -------
    mean_post : float
        Posterior mean 
    var_post : float
        Posterior variance
    """

    # get posterior mean
    mean_post = mean_prior*var_sample/(var_sample + var_prior) + \
                mean_sample*var_prior/(var_sample + var_prior)
    
    # get posterior variance
    var_post = (1/var_prior + 1/var_sample)**(-1)

    return(mean_post, var_post)


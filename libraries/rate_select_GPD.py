# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:12:00 2023

@author: Tobias Kallehauge
"""

import numpy as np
from gibbs_GPD import gibbs_GPD
from scipy.stats import genpareto, norm, gaussian_kde
from freq_GPD import get_conf_I

def rate_select_GPD(eps, zeta, C = None, method = 'freq', target = 'pcr',
                    gibbs_params = {}, **kwargs):
    """
    Select communication rate conservatively based on observed values of 
    instantanious channel capacity based on tail approximation of generalized
    parato distribution (GPD). Various options are available. 
    
    Parameters
    ----------
    eps : float
        Target outage probability
    zeta : float
        Probability to select the threshold based on the quantile u, such that
        P(C <= u) = zeta. 
    C : np.ndarray, optional
        Observed values of capacity. If None, prior information must be 
        supplied. C should be in linear domain - i.e. bits/sec/hertz
        The default is None. 
    method : str, optional
        How to select the rate. Options are:
            freq: Frequentist approach selecting the rate only based on the 
            available samples and not prior information. 
            bay : Bayesian approach where prior information is used. 
        The default is freq
    Target: str, optional
        Which criteria to select the rate based on. Options are:
            mode : Select the unbiased consistent estimator, 
            for example the MAP estimate for bayesian.
            pcr : Select rate based on probabily correct reliability (pcr) with
            given confidence delta
        The default is pcr. 
    gibbs_params : dict
        Dictionary with parameters for gibbs sampler (see gibbs_GPD for 
        additional information). 
        
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
    
    if target == 'pcr':
        assert 'delta' in kwargs, \
            'Supply confidence parameter delta for prc rate selection'
        
    n = len(C)
    
    if method == 'freq' and n > 1/zeta:

        # find the threshold based on zeta
 
        r = int(np.ceil(zeta*n))
        C_ord = np.sort(C)
        u = C_ord[r - 1] # subtract 1 since counting from zero
        Y = u -  C[C < u] # observations below u
        
        # get maximum likelihood estiamtes of the parameters
        xi_est,_, sigma_est = genpareto.fit(Y, floc = 0)
        p_u_est = r/n # maximum liklelihood estimate
        q_est = u - (sigma_est/xi_est)*((p_u_est/eps)**xi_est - 1)
        
        # select rate 
        if target == 'mode':
            R = q_est
        else: # target == 'pcr'
        
            # chose R as the lowest value in confidence interval based on assymptotic normality of q_est estimate. 
            delta = kwargs['delta']
                
            # choose test used to get confidence interval (default is Wald)
            freq_test = kwargs.get('freq_test', 'Wald')
            
            R = get_conf_I(delta, q_est, sigma_est, xi_est, u, p_u_est, 
                           Y, n, eps,test_type = freq_test)                
                
            if np.isnan(R): # due to numerical inaccuracies when n is small compared to 1/zeta
                R = 0
                
    if method == 'freq' and n <= 1/zeta: # not sufficient data to estiamte GPD 
        R = 0
                
    elif method == 'bay':
        
        assert 'prior' in kwargs, \
            'Provide prior parameters for Bayesian framework'
            
        if n <= 1/zeta: # not sufficient data
            q_mean, q_var = kwargs['prior']['prior_q'] # prior mean and variance for quantile (in log domain)
            
            if target == 'mode':
                R_ln = q_mean
            else: # target = 'pcr'
                R_ln =  norm.ppf(kwargs['delta'],
                                     loc = q_mean, 
                                     scale = np.sqrt(q_var))

            R = np.exp(R_ln)
            
        
        else: 
            
            # setup some options
            assert 'T' in gibbs_params, \
                'Provide length T of MCMC chain for Bayesian method'
            burn_in = gibbs_params.get('burn_in', 1000)
            
            
            theta_q = kwargs['prior']['prior_q']
            theta_xi = kwargs['prior']['prior_xi']    
            
            gibbs = gibbs_GPD(C, 
                              theta_q = theta_q, 
                              theta_xi = theta_xi,
                              eps = eps, 
                              zeta = zeta,
                              log_q = gibbs_params.get('log_q',True))
            
            # standard deviations for proposal distribution in MCMC simulator
            std_q = np.sqrt(theta_q[1])*2 
            std_xi = np.sqrt(theta_xi[1])*3 
            std_zeta_u = gibbs.F_zeta_prior.std()*3 
            
            # simulate posterior distribtuion with Metropolis within Gibbs
            q_gibbs, xi_gibbs, zeta_u_gibbs, accept_prop = gibbs.sim(std_q = std_q, 
                                                    std_xi = std_xi,
                                                    std_zeta_u = std_zeta_u,
                                                    T_calibrate = burn_in + 500,
                                                    **gibbs_params)
            
            # remove samples before burn_in
            q_gibbs = q_gibbs[burn_in:]
            xi_gibbs = xi_gibbs[burn_in:]
            zeta_u_gibbs = zeta_u_gibbs[burn_in:]
            
            if target == 'mode':
                # estimate mode of q_qibbs with gaussian kernel
                q_rng = np.linspace(q_gibbs.min(), q_gibbs.max(),1000)
                f = gaussian_kde(q_gibbs)(q_rng)
                R = q_rng[f.argmax()]
            else:
                R = np.quantile(q_gibbs, kwargs['delta'])


    return(R)

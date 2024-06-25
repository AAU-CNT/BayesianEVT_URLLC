# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:09:14 2023

@author: Tobias Kallehauge
"""

import numpy as np
from scipy.stats import norm, genpareto, chi2
from scipy.optimize import minimize_scalar, bisect

def get_conf_I(alpha,q_eps, sigma, xi, u, p_u,Y, n, eps, one_sided = True,
               test_type = 'Wald'):
    """
    Computes confidence interval for epsilon quantile q of observation by 
    utilizing extreme value theory of extrapolating tail based on estimated
    generalized pareto distribution (GPD) parameters for some threshold and 
    estimated probability of beeing below that threshold. The confidence
    interval relies on either assymptotic normality of maximum likelihood 
    estiamtes (Wald test) or using profile likelihood (Walks theorem)

    Parameters
    ----------
    alpha : float
        Confidence parameter, the confidence is 1 - alpha
    q_eps : float
        Estimated epsilon-quantile
    sigma : float
        Estimated scale parameter for GPD
    xi : float
        Estimated shape parameter for GPD 
    p_u : float
        Estimated probability of beeing below threshold
    Y : np.ndarray
        Observations below threshold. 
    n : int
        How many observations of the random variable (above and below u)
    eps : float
        probability for quantile
    one_sided : bool, optional
        If the confidence interval should be one sided. If True, the upper 
        bound is probided (lower bound is - infty). If false, a two-sided 
        interval is provided. The default is True.

    Returns
    -------
    I : float or tuple
        If one_sided, the upper bound of the confidence interval. If false
        a tuple of lower, upper bound is provided. 
    """
    
    if test_type == 'Wald':
        I = Wald_test(alpha, q_eps, sigma, xi, p_u, Y, n, eps, one_sided)
    elif test_type == 'profile': # based on log-profile
        I = profile_like(alpha,q_eps, xi, u, p_u, Y, eps)
    return(I)
    
def Wald_test(alpha,q_eps, sigma, xi, p_u,Y, n, eps, one_sided):
    """
    Computes confidence interval based on Wald test 
    
    See get_conf_I for parameter description. 
    """
    # get variance of p_u following a binomial distribtion with parameters r,n
    # V_p_u = beta.var(a = r, b = n + 1 - r) 
    V_p_u = p_u*(1-p_u)/n
    
    # get variance of sigma and xi in GPD from observed information
    j_sigma_xi = obs_info(sigma,xi, Y)
    V = np.linalg.inv(j_sigma_xi)
    V_extended = np.array([[V[0,0], V[0,1],     0],
                           [V[1,0], V[1,1],     0],
                           [0     ,      0, V_p_u]])
    
    # use delta method to compute variance of q_eps
    g = get_gradient(sigma,xi,p_u,eps)
    V_q = g @ V_extended @ g
    if V_q <= 0: # can happen due to numerical errors
        return(np.nan)
    
    if one_sided:
        I = norm.ppf(alpha, loc = q_eps, scale = np.sqrt(V_q))
    else:
        I = norm.interval(1 - alpha, loc = q_eps, scale = np.sqrt(V_q))
        
    return(I)

def profile_like(alpha,q_eps, xi, u, p_u, Y, eps):
    """
    Computes confidence interval based on profile likelihood confidence 
    intervals which are based on Wilks theorem. 
    
    See get_conf_I for parameter description. 
    """
    
    # get log-likelihood at estimated parameters
    L_max = GPD_q_loglike(Y, xi, u, p_u, q_eps, eps)
    
    # set bounds for searching for xi in profile-loglikelihood
    bounds = [xi - 2, xi + 2]
    
    # get critical value in chi2 distribtion
    X_alpha = chi2.ppf(1-alpha, df = 1)
    
    # Now search for the minium value in the confidence interval as a function
    # of q as lower intersection between chi2 statistic and critical value
    f_obj = lambda q : get_chi2_stat(Y,q,u,p_u,eps,bounds,L_max) - X_alpha

    try:
        # search for minimum value between 0 and estimated q
        I_min = bisect(f_obj, 0, q_eps)
    except:
        I_min = 0
        
    return(I_min)

# =============================================================================
# Various helper functions for Wald test
# =============================================================================

def obs_info(sigma, xi, Y):
    """
    Computes observed information matrix of generalized Pareto distributuion
    given observations Y.

    Parameters
    ----------
    sigma : float
        Postive scale parameter
    xi : float
        Shape parameter
    Y : np.ndarray
        Observations Y 

    Returns
    -------
    j : np.ndarray
        2 x 2 array with observed information. 

    """
    
    j = np.zeros((2,2))
    
    # some tempary vaiable
    a = sigma/(Y.copy()*xi) # copy Y since values are changed when a is close to -1
    
    # when a ~= -1, there are numerical problems so subtract a small number
    a[np.abs(a + 1) < 1e-12] -= 1e-14
    
    # second derivative with respect to sigma
    d_sigma2 = -(-len(Y)+(1+1/xi)*((2*a + 1)/((a + 1)**2)).sum())/sigma**2
    
    # second order derivative with respect to xi
    term1 = 2*np.log(1 + 1/a).sum()
    term2 = -(2/(1 + a)).sum()
    term3 = -(1 + xi)*(1/(1 + a)**2).sum()
    d_xi2 = -(term1 + term2 + term3)/xi**3
    
    # cross term with spect to xi and sigma
    term1 = (1 + 1/xi)*(((sigma/Y))/(a + 1)**2).sum()
    term2 = -(1/(a + 1)).sum()
    d_sigma_xi = (term1 + term2)/(sigma*xi**2)
        
    # observed information is negative of the Hessian
    j = -np.array([[d_sigma2  , d_sigma_xi],
                   [d_sigma_xi, d_xi2     ]])
    
    return(j)
    

def get_gradient(sigma, xi, pu, eps):
    """
    Compute Jacobian of reparametrisation with from (sigma,xi,pu) -> (q) when
    
    q_eps = u - (sigma/xi)*((pu/eps)**xi - 1)
    """
    
    a = pu/eps
    d1 = (-1/xi)*(a**xi - 1) 
    d2 = (sigma/xi)*(a**xi*(1/xi - np.log(a)) - 1/xi)
    d3 = -(sigma/pu)*a**xi
    
    g = np.array([d1,d2,d3])
    
    return(g) 

# =============================================================================
# Various helper functions for profile likelihood
# =============================================================================
def GPD_q_loglike(Y, xi, u, p_u, q, eps):
    """
    Evaluated generalized pareto distribution loglikelihood in a 
    re-parameterized version.
    

    Parameters
    ----------
    y : np.ndarray
        Input to loglikelihood
    xi : float
        Shape parameter
    u : float
        Chosen threshold
    p_u : float
        Probability that the RV X is below u, is.e., P(X < u)
    eps : float
        Probability of quantile q
    q : float
        Quantile for probability eps, i.e. P(X < q) = eps
    
    Returns
    -------
    f : np.ndarray
        Evauated logpdf
    """
    
    sigma = (u - q)*xi/((p_u/eps)**xi - 1)
    
    l_y = genpareto.logpdf(Y, c = xi, scale = sigma) # vector for e    
    mask = ~np.isinf(l_y) # filter away when 1 + xi*y/sigma <= 0 (evaluated to -inf in numpy)
    l = l_y[mask].sum()
    return(l)

def GPD_profile(Y, q, u, p_u, eps, bounds):
    """
    Profile likelihood with respect to q of the re-prametrized version
    """
    
    # set objective function
    f_obj = lambda xi : - GPD_q_loglike(Y, xi, u, p_u, q, eps)
    
    res = minimize_scalar(f_obj, bounds = bounds) # cound be nan if no succes
    
    if res.success:
        return(GPD_q_loglike(Y, res.x, u, p_u, q, eps))
    
    else:
        return(np.nan)
    
def get_chi2_stat(Y, q, u, p_u, eps, bounds, L_max):
    """
    Get chi2 statistic, which follows a chi2 distribution given the correct 
    parameter for q. 
    """
    
    # set objective function
    f_obj = lambda xi : - GPD_q_loglike(Y, xi, u, p_u, q, eps)
    
    res = minimize_scalar(f_obj, bounds = bounds) # cound be nan if no succes
    
    if not res.success:
        return(np.nan)
    
    L_P_q = GPD_q_loglike(Y, res.x, u, p_u, q, eps)
    
    chi2_stat = 2*(L_max - L_P_q)
    
    return(chi2_stat)
    
    
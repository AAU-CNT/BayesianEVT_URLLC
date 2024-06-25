# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:24:15 2023

@author: Tobias Kallehauge
"""

import numpy as np
from scipy.stats import norm, genpareto, beta, lognorm
import matplotlib.pyplot as plt


class gibbs_GPD:
    """
    Class for Gibbs sampling of the posterior distribution for paramerters for 
    the Generalized pareto distribution (GPD). The distribution is assumed for
    the lower tail of the observed variable. The threshold for the GPD is 
    assumed to be equal or less than a quantile with certain probability zeta.
    The gibbs sampler is designed for simulating from the quantile with 
    probability eps < zeta. The prior information is on the eps-quantile and
    GPD shape xi. 
    
    See [arXiv link] for a mathematical brackground.
    """
    
    def __init__(self, X, theta_q, theta_xi, eps, zeta, log_q = True):
        """
        Intializes Gibbs sampler

        Parameters
        ----------
        X : np.ndarray or list
            Observed valus (where lower tail is assumed to follow a GPD)
        theta_q : np.ndarray or list
            Lenght 2 array with prior parameters : [mean, var] for the
            eps-quantile q of the variable X such that the prior for q is 
            normal specified mean and var
        theta_xi : np.ndarray or list
            Similar to theta_q, but for the shape parameter xi (for X in linear domain)
        eps : float
            Probability corresponding to theta_q
        zeta : float
            Probability such that the exedances below the zeta-quantile of X
            follows a GPD
        log_q : bool, optional
            If True, the prior on q is of log(q) rather than q in which case
            q is lognormal rather than normal. 
        """

        # set parameters
        self.theta_q = theta_q
        self.theta_xi = theta_xi
        self.eps = eps
        self.zeta = zeta
        
        # setup some variables
        self.n = len(X)
        # check if n is sufficiently large to use the data X
        if self.n <= 1/self.zeta:
            raise(ValueError(f'The number of sample is insufficient for the GPD to apply, for zeta = {zeta:.1e}, at least {int(1/zeta) + 1} samples are required'))
            
            
        # setup threshold variable
        X_ord = np.sort(X)
        self.r = int(np.ceil(self.zeta*self.n)) # the order statistic number - int rounds down
        self.u = X_ord[self.r - 1] # subtract 1 since counting from zero
        
        # deficit below u
        self.Y = self.u - X[X < self.u]
            
        # setup priors 
        
        # eps-quantile
        if log_q: # lognormal prior
            # see scipy documentation for parametrization
            self.F_q_prior = lognorm(scale = np.exp(theta_q[0]), # scale parameter shoud be exp(mu)
                                     s = np.sqrt(theta_q[1])) # s parameters should be std = sqrt(var)
        else: # normal prior    
            self.F_q_prior = norm(loc = theta_q[0], 
                                  scale = np.sqrt(theta_q[1]))
        
        # shape parameter
        self.F_xi_prior = norm(loc = theta_xi[0],
                               scale = np.sqrt(theta_xi[1]))
        
        # zeta_u : P(X <= u) ~ Beta(r, n + 1 - r)
        self.F_zeta_prior = beta(a = self.r, 
                                 b = self.n + 1 - self.r)
            

            
    def sim(self, T, std_q, std_xi, std_zeta_u,  q_init = None, xi_init = None,
            show_trace = False, calibrate_proposal = False,
            max_iter_calibrate = 10, T_calibrate = 400, burn_in = 100, 
            verbose = True, 
            **kwargs):
        """
        Simulate from posterior distriubution using Metropolis within Gibbs.
        The full conditional for q is known based on the data. 
        The proposal densities for xi and sigma is a symmetric Gaussian 
        proposal density.
        
        Note that the initial guess for zeta_u = P(X <= u) is the mode of its
        prior distribution (r-1)/(n-1)

        Parameters
        ----------
        T : int
            Length of MCMC chain 
        std_q, std_xi, std_zeta_u : float
            Proposal standard deviation for q, xi and zeta_u
        show_trace : bool
            If True, plot the trace of q, xi and zeta_q after the simulation.
            The default is False.
        calibrate_proposal : bool, optional
            If True, rerun chain with new proposal standard deviation untill 
            acceptance probability for all parameters is within 20-50%. 
            The default is False.
        max_iter_calibrate : int, optional
            Maxium number of itrations in calibration of the proposal 
            distributions. The default is 10
        T_calibrate : int, optional
            After how many iterations to check if acceptance probability is 
            within the desired range of 20-50%. 
        burn_in : int, optional
            Burn in period: How many initial samples to discard when 
            calculating acceptance probability. Also used in the traceplot.
            The default is 100.
        kwargs : dict
            Various settings for plotting and recalibration

        Returns
        -------
        q : np.ndarray
            Simulated q parameters (after burn in)
        xi : np.ndarray
            Simulated xi parameters (after burn in)
        sigma: np.ndarray
            Simulated sigma parameters (after burn in)
        accept_prop_hist : np.ndarray
            Acceptance ratio for the three parameters (q, xi, zeta_u)
        """
        
        # setup return_arguments
        q = np.zeros(T)
        xi = np.zeros(T)
        zeta_u = np.zeros(T)
        accept_prop_hist = np.ones((T,3), dtype = 'bool')
    
        
        # setup initial values
        if q_init is None:
            # sample from prior distribution, but never higher than u
            q_init = min(self.F_q_prior.rvs(), self.u*.99)
        q[0] = q_init
        
        if xi_init is None:
            xi_init = self.F_xi_prior.rvs()
        xi[0] = xi_init
        
        zeta_u[0] = (self.r - 1)/(self.n - 1) # mode of beta distribution
                
        for t in range(1, T):
# =============================================================================
# draw q
# =============================================================================
            q_proposal = norm.rvs(loc = q[t-1], scale = std_q)
            
            if q_proposal >= self.u*.99 or q_proposal <= 0:
                # too close to u will cause negative variance in likelihood
                # negative value will casue prior pdf to be zero
                accept = False
            else:
                # get Metropolis ratio
                llike_q_proposal = self.GPD_q_loglike(self.Y, xi[t-1], self.u,
                                                      zeta_u[t-1], q_proposal,
                                                      self.eps)
                if np.isinf(llike_q_proposal):
                    # due to y values above possible maximum with new parameters
                    accept = False
           
                else: 
                        
                    llike_q_current = self.GPD_q_loglike(self.Y, xi[t-1], self.u,
                                                         zeta_u[t-1], q[t-1],
                                                         self.eps)
                
                    prior_q_proposal = self.F_q_prior.logpdf(q_proposal)
                    prior_q_current =  self.F_q_prior.logpdf(q[t-1])
                    
                    
                    MH_log = llike_q_proposal + prior_q_proposal \
                            - llike_q_current - prior_q_current
                    
                    A = min(MH_log, 0) # log acceptance probability
            
                    if np.log(np.random.random()) <= A:
                        accept = True
                    else:
                        accept = False
            
            if accept:                     
                q[t] = q_proposal
            else:
                q[t] = q[t-1]
                accept_prop_hist[t,0] = False

# =============================================================================
# draw xi
# =============================================================================
            xi_proposal = norm.rvs(loc = xi[t-1], scale = std_xi)
            
            # get Metropolis ratio
            llike_xi_proposal = self.GPD_q_loglike(self.Y, xi_proposal, self.u,
                                                  zeta_u[t-1], q[t], self.eps)
            if np.isinf(llike_xi_proposal):
                # due to y values above maximum with new parameters
                accept = False
                
            else:    
                llike_xi_current = self.GPD_q_loglike(self.Y, xi[t-1], self.u,
                                                      zeta_u[t-1], q[t], self.eps)
                
                
                
                prior_xi_proposal = self.F_xi_prior.logpdf(xi_proposal)
                prior_xi_current =  self.F_xi_prior.logpdf(xi[t-1])
                
                MH_log = llike_xi_proposal + prior_xi_proposal\
                        - llike_xi_current - prior_xi_current
                
                A = min(MH_log, 0) # log acceptance probability
        
                if np.log(np.random.random()) <= A:
                    accept = True
                else:
                    accept = False
                
            if accept: 
                xi[t] = xi_proposal
            else:
                xi[t] = xi[t-1]
                accept_prop_hist[t,1] = False
                
# =============================================================================
# draw zeta_u
# =============================================================================
            
            zeta_u_proposal = norm.rvs(loc = zeta_u[t-1], scale = std_zeta_u)
            
            if zeta_u_proposal < self.eps: # must be larger than epsilon
                accept = False
                
            else: 
                # get Metropolis ratio
                llike_zeta_u_proposal = self.GPD_q_loglike(self.Y, xi[t], self.u,
                                                  zeta_u_proposal, q[t], self.eps)
                
                if np.isinf(llike_zeta_u_proposal):
                    # due to y values above maximum with new parameters
                    accept = False
                    
                else:
                    llike_zeta_u_current = self.GPD_q_loglike(self.Y,
                                                              xi[t], 
                                                              self.u, 
                                                              zeta_u[t-1], 
                                                              q[t], self.eps)
                    
                    prior_zeta_u_proposal = self.F_zeta_prior.logpdf(zeta_u_proposal)
                    prior_zeta_u_current =  self.F_zeta_prior.logpdf(zeta_u[t-1])
                    
                    MH_log = llike_zeta_u_proposal + prior_zeta_u_proposal\
                            - llike_zeta_u_current - prior_zeta_u_current
                    
                    A = min(MH_log, 0) # log acceptance probability
                
                    if np.log(np.random.random()) <= A:
                        accept = True
                    else:
                        accept = False
                    
            if accept:
                zeta_u[t] = zeta_u_proposal
            else:
                zeta_u[t] = zeta_u[t-1]
                accept_prop_hist[t,2] = False
            
# =============================================================================
# Run calibation     
# =============================================================================
    
            if calibrate_proposal and t == T_calibrate:
                assert burn_in < T_calibrate
                
                # get current acceptance probability 
                accept_prop_cali = np.mean(accept_prop_hist[burn_in:t], 
                                           axis = 0)
                
                # get number of recalibrations so far
                N_calibrate = kwargs.get('N_calibrate', 0) # default to 0
                          
                
                # at least one acceptance probability is not within range
                if ((accept_prop_cali < 0.2) | (0.5 < accept_prop_cali)).any()\
                    and N_calibrate < max_iter_calibrate : 
                        
         
                        
                    kwargs['N_calibrate'] = N_calibrate + 1
               
                    
                    std = [std_q, std_xi, std_zeta_u]
                    
                    if verbose: 
                        tmp = ', '.join([f'{i*100:.0f}%' for i in accept_prop_cali])
                        print(f'Calibrating {N_calibrate} of {max_iter_calibrate}, acceptance probabilites: {tmp}.')
                    
                    for i in range(3):
                        if accept_prop_cali[i] < 0.2: # to small acceptance
                            std[i] *= 0.5  # decrease acceptance probability
                        elif accept_prop_cali[i] > 0.5: # to large acceptance
                            std[i] *= 2  # increase acceptance probability
                        
                    # rerun MCMC chain with new proposal densities
                    q, xi, zeta_u, accept_prop = \
                        self.sim(T, std[0], std[1], std[2],
                                 burn_in = burn_in, 
                                 show_trace = show_trace, 
                                 calibrate_proposal = True,
                                 max_iter_calibrate = max_iter_calibrate,
                                 T_calibrate = T_calibrate,
                                 verbose = verbose,
                                 **kwargs)
                    
                    
                    return(q, xi, zeta_u, accept_prop)
                    
                    
            # if within range just continue simulations
                
                
                
        
        # compute acceptance probabilites after burn 
        accept_prop = np.mean(accept_prop_hist[burn_in:], axis = 0)

# =============================================================================
# Plot
# =============================================================================
        
        if show_trace:
            
            remove_burn_in_trace = kwargs.get('remove_burn_in_trace', False)
            if remove_burn_in_trace:
                rng = np.arange(burn_in, T)
            else:
                rng = np.arange(T)
                
            show_mean = kwargs.get('show_mean', True)
            
            show_true = kwargs.get('show_true', False)
            
            fig, ax = plt.subplots(nrows = 3, figsize = (6,6))
            # q
            ax[0].plot(rng,q[rng])
            ax[0].axvline(burn_in, c = 'r')
            ax[0].set_title(r'q for $\epsilon = $' + f'{self.eps:.1e}, acceptance probability {accept_prop[0]*100:.1f}%')
            ax[0].set_xticks([])
            # xi
            ax[1].plot(rng,xi[rng])
            ax[1].axvline(burn_in, c = 'r')
            ax[1].set_title(r'$\xi$' + f': acceptance probability {accept_prop[1]*100:.1f}%')
            ax[1].set_xticks([])
            # sigma
            ax[2].semilogy(rng,zeta_u[rng])
            ax[2].axvline(burn_in, c = 'r')
            ax[2].set_title(r'$\zeta_u$' + f': acceptance probability {accept_prop[2]*100:.1f}%')
            
            if show_mean:
                ax[0].axhline(q[burn_in:].mean(), c = 'C2', label = 'Mean')
                ax[1].axhline(xi[burn_in:].mean(), c = 'C2')
                ax[2].axhline(zeta_u[burn_in:].mean(), c = 'C2')
                
            if show_true:
                assert 'true_vals' in kwargs
                for i in range(3):
                    ax[i].axhline(kwargs['true_vals'][i], c= 'C1', label = 'True')
                ax[0].legend()
            
            if kwargs.get('save', False):
                fig.savefig(kwargs.get('save_path', 'traceplot.png'), dpi = 400)
            
            plt.show()
            
        
        return(q, xi, zeta_u, accept_prop)

    def GPD_q_loglike(self, y, xi, u, zeta_u, q, eps):
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
        zeta_u : float
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
        
        sigma = (u - q)*xi/((zeta_u/eps)**xi - 1)
        
        l = genpareto.logpdf(y, c = xi, scale = sigma).sum()
        return(l)
            
            
        
            
            
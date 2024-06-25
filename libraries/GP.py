# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:09:21 2021

@author: Tobias Kallehauge
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.gaussian_process.kernels import Matern
from scipy.misc import derivative
from scipy.linalg import cholesky

class GP:
    """
    Class for model selection, prediction and statistical analysis of gaussian
    processes (GP)
    
    Functionality:
        - Fit hyperpameters for mean and covariance function with noisy 
        observations
        - Predictive distribution given estimated hyper parameters
    """
    
    def __init__(self, mean = 'zero', cov = 'Gudmundson', x_dim = 2,
                 N_mean_para = None, mean_func = None):
        """
        Setup mean (m) and covariance function (C) such that GP(m,C) is the 
        gaussian process. 

        Parameters
        ----------
        mean : str, optional
            Mean function for GP. The options are:

                'zero' : No explicit mean function modelled. Use this when no
                information about mean is known.
                'constant':  Similar to zero by allow for non-zero mean. 
                
                'linear' : Mean is generalised linear function of input x i.e.
                m(x;beta) = h^T(x)beta where h is any function and 
                beta are the parameters. Use the keyword parameters
                N_mean_para and mean_func parameters to set the 
                number of parameters and particular mean function. 
                
            The default is 'zero'.
        cov : str, optional
            Covariance function. The options are:
                
                'Gudmundson': C(s,s') = var*exp(-||x1  - x2||/(d_c))
                where var is the variance and d_c is the correlation distance.
                The l2-norm is used in the eksponential. 
            Use a dictionary with covariance parameters
            The default is 'Gudmundson'.
        x_dim : int, optional
            Dimension of the input x. 
            The default is 2. 
        N_mean_para : int, optional
            Number of parameters when using 'linear' mean function. 
        mean_func : callable, optional
            Mean function of input when using 'linear' mean function. 
            Should be vectorised able to handle multible inputs. 
        """
        
        assert isinstance(x_dim,int), 'x_dim should be an integer'
        self.x_dim = x_dim
        self.N_mean_para = N_mean_para
        
        # setup mean function
        self.mean = mean
        if self.mean == 'zero':
            self.m_x = lambda x, *args: np.zeros(x.shape[0])
        elif self.mean == 'const':
            self.N_mean_para = 1
            self.m_x = lambda x, c : np.repeat(c, x.shape[0])
            self.mean_func = lambda x : np.ones((x.shape[0],1))
        elif self.mean == 'linear':
            if 'N_mean_para' == None or mean_func  == None:
                msg1 = 'N_mean_para and mean_func should be specified using '
                msg2 = 'linear mean'
                raise(ValueError(msg1 + msg2))
            # test mean function
            try:
                x_test = np.random.random((10,self.x_dim))
                if self.x_dim == 1:
                    x_test = x_test.flatten()
                h = mean_func(x_test)
                assert h.shape == (10,self.N_mean_para)
            except AssertionError:
                msg = 'Mean function does not work or has wrong dimension'
                raise(ValueError(msg))
            
            # setup function if test passed
            self.mean_func = mean_func
            self.m_x = lambda x, beta: mean_func(x).dot(beta)
            self.N_mean_para = N_mean_para
        
        # setup covariance function
        self.cov = cov
        if self.cov == 'Gudmundson':
            self.C_x = self.cov_gudmundson
            self.N_cov_para  = 2
        elif self.cov == 'exp':
            self.C_x = self.cov_exp
            self.N_cov_para  = 2
        elif self.cov == 'Matern':
            self.C_x = self.cov_matern
            self.N_cov_para = 3
        else:
            raise(ValueError('Chose a valid covariance function!'))
        
        
    #%% Covariance and related functions
    def cov_gudmundson(self,x1,x2,d_c,var):
        """
        Covariance matrix with Gudmundson model for location vectors x1,x2.
        
        We have : C(s,s') = var*exp(-||x1 - x2||/(d_c))
    
        Parameters
        ----------
        x1, x2 : ndarray
            Location vectors. 
        d_c : float
            Correlation discance parameter
        var : float
            Variance parameters
    
        Returns
        -------
        C : np.ndarray
            Covariance matrix for locations. 
    
        """
        d = self._get_dist(x1,x2)
            
        C = var*np.exp(-d/d_c)
        return(C)
    
    def cov_exp(self,x1,x2,d_c,var):
        """
        Covariance matrix with exponential model for location vectors x1,x2.
        
        We have : C(s,s') = var*exp(-||x1 - x2||^2/(d_c))
    
        Note that exp has the norm squared where gudmundson does not.             
        
        Parameters
        ----------
        x1, x2 : ndarray
            Location vectors. 
        d_c : float
            Correlation discance parameter
        var : float
            Variance parameters
    
        Returns
        -------
        C : np.ndarray
            Covariance matrix for locations. 
    
        """
        d = self._get_dist(x1,x2)**2
            
        C = var*np.exp(-d/d_c)
        return(C)
    
    def cov_matern(self,x1,x2,d_c,nu,var):
        """
        Matérn covariance function. 
        See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
        

        Parameters
        ----------
        x1, x2 : ndarray
            Location vectors.
        d_c : float
            Correlation distance. Should be postive
        nu : float
            Smoothness parameter 1/2 is rough (cov-gudmondson) and infty is 
            exponential 
        var : float
            Scale parameter. Should be positive

        Returns
        -------
        C : ndarray
            Covairance matrix for locations 
        """
        
        k = var * Matern(length_scale=d_c,nu = nu)
        C = k(x1,x2)
        
        return(C)
        
    
    
    def grad_C(self,para, x, noise):
        """
        Computes gradtion of covariance function with respect to the
        appropriate parameters with and without considering noise. 
    
        Parameters
        ----------
        para : ndarray
            Parameters for covriance model.
        x : ndarray
            Input vectors
        noise : bool
            If the gradient should be compute with respect to the noise 
            variance. 
        
        Returns
        -------
        grad : ndarray
            Gradient where axis 0 is parmeters and axis 1,2 are x. 
    
        """
        # Compute gradient depending on the covariance model. 
        
        if self.cov in ('Gudmundson', 'exp'):
            # get some variables
            d_c, var_f = para
            d = self._get_dist(x,x)
            if self.cov == 'exp':
                d = d**2 # exponential has the squared norm
            tmp1 = d/d_c
            tmp2 = np.exp(-d/d_c)
            n = x.shape[0]
            
            # make gradient
            if noise:
                N_para = 3
            else:
                N_para = 2
            grad = np.zeros((N_para,n,n))
            
            grad[0] = var_f*tmp1*tmp2/d_c
            grad[1] = tmp2
            
            if noise:
                grad[2] = np.eye(n)
            
        if self.cov == 'Matern':
            d_c, nu, var_f = para
            
            # Setup functions for numerical derivartives with respect to 
            # length scale and smoothness 
            k_d_c = lambda d_c: self.C_x(x,x,d_c,nu,var_f)
            k_nu = lambda nu: self.C_x(x,x,d_c,nu,var_f)
            n = x.shape[0]
            
            # make gradient
            if noise:
                N_para = 4
            else:
                N_para = 3
                
            grad = np.zeros((N_para,n,n))
            
            grad[0] = derivative(k_d_c,d_c,dx = 1e-6,order = 9)
            grad[1] = derivative(k_nu,nu,dx = 1e-6,order = 9)
            grad[2] = self.C_x(x, x, d_c, nu, 1)
            
            if noise:
                grad[3] = np.eye(n)
             
        return(grad)
    
    # %% probability distribtions (log-likelihoods, pdfs, etc.) 
    
    def log_marginal(self,x,y,para_cov, para_mean = None, var_n = None):
        """
        The (log) maginal distribution of y|x, para_mean, para_cov, var_n which
        is marginalised over the gaussian process when noise is observed. When 
        there is no noise, it is simply the log likelihood. 
    
        Parameters
        ----------
        x : ndarray
            Input
        y : ndarray
            Observed output with or without noise. 
        para_cov : ndarray
            Parameters for covariance function.
        para_mean : nparray, optional
            Parameters for mean function if any for the model. 
        var_n : float or ndarray, optional
            Noise variance. If none, assume observations are noiseless.
            
        Returns
        -------
        l_mag : ndarray
            Log marginal at (x,y)
        
        """
        
        # Get mean and covariance
        mu = self.m_x(x, para_mean)
        Sigma = self.C_x(x,x,*para_cov)
        
        # add noise given noise observations
        if not var_n is None:
            if isinstance(var_n, float):
                var_n = np.repeat(var_n,x.shape[0])
            Sigma += np.diag(var_n)
        
        # Compute margianl using cholesky factorisation
        L = np.linalg.cholesky(Sigma)
        # Use least squares which should with cholesky be numeircally stable 
        alpha = self._lstsq(L.T,self._lstsq(L,y - mu))
        
        # Compute log maginal 
        l_mag = -0.5*((y-mu).dot(alpha)) - np.log(np.diag(L)).sum() 
        
        return(l_mag)
    
    def grad_log_marginal(self,x,y,para_cov, para_mean = None,var_n = None,
                          var_n_known = False):
        """
        Computes gradient with respect to covariance parameters. Also gets 
        gradient for noise variance when prompted. 
        

        Parameters
        ----------
        See log_marginal
        
        Returns
        -------
        grad : ndarray
            Gradient. 

        """
        
        # Get mean and covariance
        mu = self.m_x(x, para_mean)
        Sigma = self.C_x(x,x,*para_cov)
        
        # add noise given noise observations
        noise = False # change if noise is included
        if not var_n is None:
            if isinstance(var_n, float):
                var_n = np.repeat(var_n,x.shape[0])
            Sigma += np.diag(var_n)
            if var_n_known: # dont compute gradient wrt. noise
                noise = False
            else:
                noise = True
        
        # compute gradient of marginal 
        Sigma_inv = np.linalg.inv(Sigma)
        grad_C = self.grad_C(para_cov, x, noise)
        alpha = Sigma_inv.dot(y - mu)
        tmp = np.matmul((np.outer(alpha,alpha) - Sigma_inv),grad_C)
        grad = 0.5*np.trace(tmp,axis1 = 1, axis2 = 2)

        return(grad)      
    
    def posterior(self,x_tst,x_trn,y_trn, 
                  para_cov, para_mean = None, var_n = None):
        """
        Returns posterior predictive distribution parameters: mean and variance
        . The posterior is the posterior g(x_tst) | x_tst, x_trn, y_trn
        where g is the gaussian process. It is assumed that the parameters are 
        known. 

        Parameters
        ----------
        x_tst : ndarray
            Test locations to get predictive distribution for. 
        x_trn : ndarray
            Training locations 
        y_trn : ndarray
            Measured gaussian process at training locations with or without 
            noise. 
            
        See log_marginal for para_cov, para_mean, var_n 
            
        Returns
        -------
        mu_tst : ndarray
            Predictive mean at test locations.
        cov_tst : ndarray
            Predictive covariance at test locations.
        """
        
        # setup noise
        if var_n is None:
            noise = False
        else:
            noise = True
            
        # get means
        m_trn = self.m_x(x_trn,para_mean)
        m_tst = self.m_x(x_tst,para_mean)
    
        # # Get covariance matricies
        N_trn = x_trn.shape[0]
        C_tst_trn = self.C_x(x_tst,x_trn,*para_cov)
        C_tst = self.C_x(x_tst,x_tst,*para_cov)
        C_trn = self.C_x(x_trn,x_trn,*para_cov)
        if noise: 
            if isinstance(var_n, float):
                var_n = np.repeat(var_n,N_trn)
            C_trn = C_trn + np.diag(var_n)
        
        # Not compute mean and covariance 
        L = np.linalg.cholesky(C_trn)
        # Use least squares which should with cholesky be numeircally stable 
        alpha = self._lstsq(L.T,self._lstsq(L,y_trn - m_trn))
        mu_tst = m_tst + C_tst_trn.dot(alpha)
        v = self._lstsq(L,C_tst_trn.T)
        C_tst = C_tst - (v.T).dot(v)
        
        return(mu_tst, C_tst)
    
    #%% Model selection 
    
    def optimize_mag(self,x,y, cov_para_init, var_n_init = None, 
                     verbose = False, var_n_known = False,
                     var_lower_bound = 0.1):
        """
        Estimate parameters for covariance and possibly mean and noise. 
        Optimization is maximises the marginal likelihood using a Gradient 
        based optimization algorithm. 
        
        The implementation currently have several repeadet calculations. 
    
        Parameters
        ----------
        x : ndarray
            Training locations 
        y : ndarray
            Training data at x (the two should be the same size)
        cov_para_init : ndarray
            Initial covariance parameters 
        var_n_init : float, optional
            If true, assume y is noisy. Otherwise, assume y is noiseless. 
        v_n_known : bool
            If True, assume var_n_init is the true variance so is not 
            optimized over. 
        verbose : float
            If optimization output should be printed
        
        Returns
        -------
        para_mean : ndarray or None
            Parameters assicoated with mean. If mean is zero this is None
        cov_para : ndarray
            Parameters associated with covariance
        var_n : float
            Noise parameter. If var_n_init is None, this is None. 
            
        """
        
        # Setup noise 
        if var_n_init is None:
            noise = False
        else:
            noise = True
            
        # Setup some constants
        # With linear mean, make design matrix 
        if self.mean in ('const', 'linear'): 
            X = self.mean_func(x)
        
        def setup_para(para):
             # split parameters into covariance para and noise variance
            if noise and (not var_n_known):
                para_cov = para[:-1]
                var_n = para[-1]
            elif noise and var_n_known:
                para_cov = para
                var_n = var_n_init
            else:
                para_cov = para
                var_n = None

            # get beta as a function of the other parameters
            if self.mean in ('const','linear'):
                
                Sigma = self.C_x(x,x,*para_cov)
        
                # add noise given noise observations
                if noise:
                    Sigma += var_n*np.eye(x.shape[0])
                
                # maximise likelihood with. mean para with normal equation
                # note that the covariance function is accounted for. 
                Sigma_inv = np.linalg.inv(Sigma)
                    
                para_mean = np.linalg.inv(X.T.dot(Sigma_inv).dot(X)) \
                                .dot(X.T).dot(Sigma_inv).dot(y)

            else:
                para_mean = None
                
            return(para_mean, para_cov, var_n)
        
        # setup objective and gradient function as local functions
        def f_g(para):
            para_mean, para_cov, var_n = setup_para(para)
         
            l_y = self.log_marginal(x,y,
                                    para_cov = para_cov,
                                    para_mean = para_mean,
                                    var_n = var_n)
            
            grad_f_y = self.grad_log_marginal(x,y,
                                          para_cov = para_cov,
                                          para_mean = para_mean,
                                          var_n = var_n,
                                          var_n_known= var_n_known)
            
            # Return negative since we are using mimization module    
            return(-l_y, -grad_f_y)

        # Setup get parameters bounds
        bound_low = []
        bound_high = []
        if self.cov in ('Gudmundson', 'exp'):
            bound_low.extend([.1,.1])
            bound_high.extend([np.inf,np.inf])
        elif self.cov == 'Matern':
            # nu = 0.5 is Gudmundson model 
            bound_low.extend([.1,0.5,0.1]) 
            bound_high.extend([np.inf,15,np.inf])
        if noise and (not var_n_known):
            bound_low.append(var_lower_bound)
            bound_high.append(np.inf)
            
        bounds = Bounds(bound_low,bound_high)
        para_init = cov_para_init
        if noise and (not var_n_known):
            para_init = np.append(para_init,var_n_init)

        # Run optimizer
        res = minimize(f_g,para_init,method = 'SLSQP', jac = True,
                        options = {'disp': verbose},
                        bounds = bounds)

        # extract parameters from solution
        para_mean, para_cov, var_n = setup_para(res.x)
        
        return(para_mean, para_cov, var_n)
    
    
    # %% Utility functions
    def _get_dist(self,x1,x2):
        """
        Get distance matrix for convariance functions. 
        """
        if self.x_dim == 1: # one dimensional computed as arrays
            d = np.abs(x1[:,None]-x2[None,:])
        else: # otherwise compute as vector
            d = np.linalg.norm(x1[:,None,:] - x2[None,:,:],axis = 2)
        return(d)
    
    @staticmethod
    def _lstsq(A,b):
        """
        Least squares estimator using numpy. 
        """
        return(np.linalg.lstsq(A,b,rcond = None)[0])
 


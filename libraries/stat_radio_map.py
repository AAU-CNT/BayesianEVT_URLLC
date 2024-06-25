# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:38:06 2023

@author: Tobias Kallehauge
"""


import numpy as np
import h5py
import pandas as pd
from functools import reduce
from tqdm import tqdm
import multiprocessing as mp
from GP import GP
from ThomasClusterRectangle import r_Thomas_discrete
from scipy.stats import gaussian_kde, genpareto

class stat_radio_map:
    """
    Bayesian estimation of fading distributiuon for ultra reliable low latency
    communication. Capability:
        - Load training dataset from file
        - Estimate fading distribtuion at each point in training dataset
        - Interpolate to get radio map. Not only for the mean, but also for 
          other fading parameters. 
        - Given observations at new location, get MAP estimate of fading 
          paramers (ML also available)
        - Get quantiles for fitted distribution
    """
    
    def __init__(self,data_index, data_path,
                 SNR_scale = 1,
                 rx_subsample = False,
                 rx_subsample_mode = None,
                 fade_subsample = 'all',
                 data_origin = 'Quadriga',
                 **kwargs):
        """
        Load data and subsample if prompted. 

        Parameters
        ----------
        data_index : int
            Pick among the datasets with this index.
        data_path : str
            Path to data folder
        SNR_scale : float, optional
            Value multiplied with SNR after loading from normalized datafile.
            The default is 1. 
        rx_subsample : bool or int, optional
            Subsampling for recievers in training set. 
            If False, no subsampling is done. If integer and rx_subsample_mode
            is 'grid' subsample every rx_subsample point. If integer and 
            rx_subsample_mode is 'random' subsample rx_subsample points 
            randomly. In the latter case rx_subsample_mode should be less than 
            the total number of recievers. 
        rx_subsample_mode, str, optional
            Subsample mode. Options are:
                grid : subsample every rx_subsample point. 
                random : draw rx_subsample points randomly (no replacement)
                list : get points at specific locations in a list. The closest 
                point available are chosen. 
                random_non_uniform : same as random but according to thomas
                process. 
        fade_subsample : str or slice, optional
            Which preprocessed data set to load from the fading data based on
            index. If all, load all data. Else, provide slice of which samples
            to load. 
        data_origin : str
            Which kind of data. Options are:
                Quadriga : Simulated from Quadriga in regular grid
                APMS : Real measured data from 127 points
            
        """
        
        # load locations 
        if data_origin == 'Quadriga':
            self.data_path = data_path +\
                             f'Stored/Distribution_map_{data_index}_radio_map.h5'
        else: # APMS data
            self.data_path = data_path + 'apms_data.h5'

        f = h5py.File(self.data_path,'r')
        
        self.s = f['ue_coordinates'][()].T
        
        self.SNR_scale = SNR_scale
        self.data_origin = data_origin
        
        if data_origin == 'Quadriga':
            # get configuration file
            self.config = pd.read_csv(data_path +
                                  f'Stored/Distribution_map_{data_index}_config.csv',
                                  index_col = 0, header=None, names = ['value'])
            # make function for extracing value based on name 
            def get_num(name) : return(eval(self.config.loc[name].value))
            self.extent = [get_num('x_min'),get_num('x_max'),
                           get_num('y_min'),get_num('y_max')]
            self.s_BS = np.array([get_num('x_tx'),
                                  get_num('y_tx'),
                                  get_num('z_tx')])
            self.N_rx = get_num('N_points_dist_map')
            self.N_rx_side = int(np.sqrt(self.N_rx))
            self.N_fade = get_num('N_samples_fading')
        else:
            self.s_BS = f['rx'][()]
            self.fading = f['fading'][()]
            self.N_rx, self.N_fade = self.fading.shape
            self.index = f['index'][()]
            
        if isinstance(fade_subsample,str):
            assert fade_subsample == 'all', 'Only subsample string mode is all'
            self.fade_reduced = False
        else:
            assert len(fade_subsample) <= self.N_fade, f'Provided slice exeeds number of samples avilable which is {self.N_fade}'
            self.fade_reduced = True
            self.fade_subsample = fade_subsample
        
        if 'coeff_real' in f: # load magnitude of actual paths
            self.coeff_mag = np.abs(f['coeff_real'][()].T + 1j*f['coeff_imag'][()].T)
            self.K = self.coeff_mag.shape[1]
            
        # Subsample if prompted
        if not rx_subsample_mode is None:
            self.subsample = True
            
            if rx_subsample_mode == 'grid':
                assert data_origin == 'Quadriga', 'Grid subsamples not supported for APMS data'
                fac = factors(self.N_rx_side - 1)
                if rx_subsample in fac:    
                    # subsample in rows and columns using a matrix of indicies
                    idx = np.arange(self.N_rx).reshape(self.N_rx_side,
                                                          self.N_rx_side)
                    # subsample index
                    idx = idx[::rx_subsample,::rx_subsample].flatten()
                       
                    self.N_rx = idx.size
                    self.N_rx_side = int(np.sqrt(self.N_rx))
                else:
                    msg = f'Can only grid sample by factors of N_rx_side,\
    choose from {fac}'
                    raise ValueError(msg)
            elif rx_subsample_mode == 'random':
                assert rx_subsample <= self.N_rx, \
                      f'rx_subsample must be less than {self.N_rx}'
                idx = np.random.choice(np.arange(self.N_rx),
                                      size = rx_subsample,
                                      replace = False)
                self.N_rx = idx.size
                
            elif rx_subsample_mode == 'list':
                N_list = len(rx_subsample)
                idx = np.zeros(N_list, dtype = 'int')
                for i in range(N_list):
                    idx[i] = np.linalg.norm((rx_subsample[i] - self.s),
                                            axis = 1).argmin()
                
                idx = np.sort(idx)
                self.N_rx = idx.size
                if data_origin == 'Quadriga':
                    del self.N_rx_side # no longer relevant
            
            elif rx_subsample_mode == 'random_non_uniform':
                assert data_origin == 'Quadriga', 'Non uniform subsamples not supported for APMS data'
                
                s_new = r_Thomas_discrete(*self.extent,rx_subsample,
                                            get_num('sample_distance'),
                                            **kwargs)
                idx = []
                # find indicies of new s
                for s in s_new:
                    idx.append(np.where((s == self.s).all(axis = 1))[0][0])
                    
                self.N_rx = len(idx)
                del self.N_rx_side # no longer relevant
            
            self.idx = np.array(idx)
            self.s = self.s[idx]
            if data_origin == 'APMS':
                self.fading = self.fading[idx]
        else:
            self.subsample = False
            
        f.close()

    def get_SNR(self, idx):
        """
        Load and return SNR values at the given indices

        Parameters
        ----------
        idx : int 
            Single index as integer or numpy array of indicies to load.
            The index values should be between 0 and self.N_rx - 1
        Returns
        -------
        W : np.ndarray
            SNR values at given index
        """
        
        assert isinstance(idx, (int,np.int32, np.int64)), 'Only integer input for allowed'
                
        if self.data_origin == 'Quadriga':
            if self.subsample:
                idx = self.idx[idx]
        
            with h5py.File(self.data_path,'r') as f:
     
                
                W = f['fading_samples'][idx]*self.SNR_scale
            
        else:
            W = self.fading[idx]*self.SNR_scale
            
        if self.fade_reduced:
            W = W[self.fade_subsample]
        
        return(W)
    
    def generate_SNR(self,idx, n):
        """
        Generate and return SNR values at the given indices by simulating based
        on path magnitues

        Parameters
        ----------
        idx : int 
            Single index as integer or numpy array of indicies to load.
            The index values should be between 0 and self.N_rx - 1
        n : int
            How many samples to generate
        Returns
        -------
        W : np.ndarray
            SNR values at given index
        """
        
        assert isinstance(idx, (int,np.int32, np.int64)), 'Only integer input for allowed'
        assert hasattr(self, 'coeff_mag'), 'Current dataset does not have the paths'
                
        if self.data_origin == 'Quadriga':
            if self.subsample:
                idx = self.idx[idx]
            
            coeff_mag = self.coeff_mag[idx]
            
            n_max = int(1e7) # limit to 10^7 samples at a time
            n_chunks = n//n_max 
            if n_chunks == 0:
                n_chunks = 1
                n_max = n
            W = []
            for i in range(n_chunks):
            
                phases = np.random.uniform(-np.pi,np.pi, size = (n_max,self.K))
                    
                SNR = np.abs((np.abs(coeff_mag[None,:])*np.exp(-1j*phases)).sum(axis = 1))**2
                W.append(SNR*self.SNR_scale)
            
            W = np.hstack(W)[:n]
            
        else:
            raise(NotImplementedError())
            
        
        return(W)
        
        
        

    def fit_map(self, variable = 'capacity', model = 'GPD', log = False,
                multiprocessing = False, N_workers = None, **kwargs):
        """
        Fit map of chosen variable to chosen distribution for the whole map.
        Note that spatial dependencies are ignored here. 
        
        Parameters are saved in object and returned
        
        Parameters
        ----------
        variable : str, optional
            Which random variable to fit a model to. Options are:
                SNR : signal-to-noise ratio (stored data contains normalized 
                                             SNR)
                capacity : instantanious capacity in bits/sec/Hz given by 
                           log2(1+SNR)
                The default value is 'capacity'
        model : str, optional
            Model for random variable. Options are:
                emp : non-parametric fitting of quantile
                GPD : generalized pareto distribution for threshold
        log : bool, optional
            If true, fit to logarithm of the variable. The default is False. 
        
        multiprocessing : bool, optional
            If True, fit using multiprocessing. The default is False.         
        
        N_workers : int, optional
            How many workers for multiprocessing if applicable. If None,
            N_workers is set to the number of CPUs on the unit. 
        
        kwargs are settings for fitting methods. The options depend on the 
        modes
        
        kwargs for emp
        --------------
        eps : float
            Probability that specifies which quantile to fit
        
        kwargs for GPD
        --------------
        zeta : float
            zeta specifies the threshold for variable under which the GDP 
            distribution is accurarte. The threshold is selected, for each 
            location as the zeta-quantile of the chosen variable.        
        
        Returns
        -------
        para : ndarray
            Parameters where each row represents a location in the dataset. 
        """
        
        if not multiprocessing:
            idx = np.arange(self.N_rx)
            self.para = self._fit_map_core(idx, model, variable, log = log,
                                           **kwargs)
            
        else:
            if N_workers is None:
                N_workers = mp.cpu_count()
                
            # Pool with progress bar
            pool = mp.Pool(processes= N_workers,
                           initargs=(mp.RLock(),), 
                           initializer=tqdm.set_lock)
            
            # setup arguments 
            idx_split = np.array_split(np.arange(self.N_rx),N_workers)

            # run multiprocessing 
            jobs = [pool.apply_async(self._fit_map_core, \
                          args = (idx_split[i], model, variable), \
                          kwds = {'log': log, 'worker_idx': i, **kwargs}) \
                          for i in range(N_workers)]
            pool.close()
            pool.join()
            
            # stack results
            mp_results = [job.get() for job in jobs]
            
            self.para = np.vstack(mp_results)
        
        return(self.para)
    
# =============================================================================
#     def _fit_scalar_hyper(self, X, cov, mean, var_n_known, **kwargs):
# =============================================================================
    
    def interpolate_map(self,s, para, cov, normalize = True, 
                        var_n_known = False, **kwargs):
        """
        Fit hyperparameters for the chosen covariance model based on fitted
        parameters. 

        Parameters
        ----------
        para : np.ndarray
            Matrix with parameters for each location.
        cov : str or list 
            Which covariacne model to use. If string, the same covariance model
            is used for all parameters. If list, then seperate models can be
            used for each location. Se GP class for options on covariance 
            functions. 
        normalize : bool or list, optional
            If True, normalize parameters. If list is povided i shold follow
            the shape of the given parameters with True/False for each 
            parameter. The default is Treue.
        var_n_known : bool, optional
            If True, assume that the variance of the parameters are known.
            Note that the keyword argumetns should then specify the known 
            variance. The default is False.
        
        Returns
        -------
        hyper : list
            List with elements as dictionaries with specified hyperparameters. 
        """
        
        
        N_para = para.shape[1]
        N_pred = s.shape[0]
        
        # Setup mean and covaiance 
        if isinstance(cov, str):
            cov = N_para*[cov]
        if isinstance(normalize, bool):
            normalize = N_para*[normalize]
            
        hyper = {}
        pred_mean = np.zeros((N_pred,N_para))
        pred_var = np.zeros((N_pred,N_para))
        for i in range(N_para):
            # fit each parameter seperately
            pred_mean_i, pred_var_i, hyper_i = \
                self._interpolate(s, para[:,i], cov[i], normalize[i], 
                                       var_n_known, **kwargs)
                
            pred_mean[:,i] = pred_mean_i
            pred_var[:,i] = pred_var_i
            hyper[i] = hyper_i
            
        return(pred_mean, pred_var, hyper)
            
        
    def _interpolate(self,s,X, cov, normalize, var_n_known = False, **kwargs):
        """
        Interpolate X to s with Gaussian processes. s should be 3 dimensional.
        
        Parameters
        ----------
        X : np.ndarray
            Should be 1 dimensional
        var_n_known : np.ndarray
            If array, assume noise variance is known
        """
        
        # extract parameters and their estimated variances
        # demean
        X = X.copy()
        X = X.flatten()
        if normalize:
            mean = X.mean()
            std = X.std()
            X = (X - mean)/std
        if var_n_known:
            var_n_init = var_n = kwargs['var_n']/std
            kwargs.pop('var_n')
        elif 'var_n_init' in kwargs:
            var_n_init = kwargs.get('var_n_init', 1)
            kwargs.pop('var_n_init')
        else:
            var_n_init = 1
            
        
        # initialize GP class
        g = GP(mean = 'zero', cov = cov, x_dim = 3)
        
        # get initial parameters
        if 'para_init' in kwargs:
            para_init = kwargs['para_init']
            kwargs.pop('para_init')
        else:
            para_init = self._get_init_para_cov(cov) 
            
        # estimate covariance parameters (noise known from data)
        _, cov_para, noise_var = g.optimize_mag(self.s, X, 
                                        cov_para_init = para_init,
                                        var_n_init = var_n_init,
                                        var_n_known = var_n_known,
                                        **kwargs)

        if not var_n_known:
            var_n = noise_var

        # save hyperparameters
        hyper = self._unpack_cov_para(cov,cov_para)
        hyper['var_n'] = var_n
        
        #  get the posterior predictive
        p_mean, p_cov = g.posterior(s, self.s, X, 
                                    para_cov = cov_para, 
                                    var_n = var_n)
        if normalize:    
            p_mean = std*p_mean + mean
            p_cov = std**2*p_cov
            
        p_var = np.diag(p_cov)
        
        return(p_mean, p_var, hyper) 
        
    def _fit_map_core(self, idx, model, variable, 
                      worker_idx = 0, log = False, **kwargs):
        """
        Static method for fitting parameters. Multiprocessing uses this. 
        """
        
        N_tasks = idx.size
        
        # Setup progress bar
        tqdm_text = "#" + "{}".format(worker_idx).zfill(3)
        progress_bar = tqdm(total = N_tasks, desc = tqdm_text, 
                            position = worker_idx, ascii = True,
                            mininterval = 1)
        

        if model in ('emp', 'emp_pdf'):
            assert 'eps' in kwargs
            eps = kwargs['eps']
            # assert 1 <= self.N_fade*eps, 'eps too small for the number of samples'

        if model == 'GPD':
            assert 'zeta' in kwargs, 'Give zeta as keyword for emp model'
            zeta = kwargs['zeta']
            assert 1 <= self.N_fade*zeta or 'N_min' in kwargs, 'zeta too small for the number of samples'
        
        # number of parameters for different models
        n_para = {'emp': 1, 'GPD': 3, 'emp_pdf': 2}

                                
        para = np.zeros((N_tasks,n_para[model]))
        with progress_bar as pbar:
            for i, idx_i in enumerate(idx):
                X = self.get_SNR(idx_i)
                if variable == 'capacity':
                    X = np.log2(1 + X)
                if log:
                    X[X <= 1e-16] = 1e-16 # for numerical stability
                    X = np.log(X)
                try:
                    if model == 'emp':
                        para[i] = quantile_est(X, eps)
                        
                    elif model == 'emp_pdf': # also compute pdf evaluated at quantile
                        X_eps = quantile_est(X, eps)
                        f_eps = gaussian_kde(X)(X_eps)[0] # pdf evaluated at X_eps
                        ln_f_esp = np.log(f_eps) # transform via log to ensure postive prediction
                        para[i] = [X_eps, ln_f_esp]
                    
                    
                    elif model == 'GPD':
                          
                        # find threshold 
                        if 'u' in kwargs:
                            u = kwargs['u'][idx_i]
                        else:
                            zeta_n = zeta
                            if 'N_min' in kwargs:
                                n = len(X)
                                if n*zeta_n <= kwargs['N_min']:
                                    zeta_n = kwargs['N_min']/n
                                
                            r = int(np.ceil(zeta_n*len(X)))
                            X_ord = np.sort(X)
                            u = X_ord[r - 1]
                    
                        # get exedances below threshold 
                        Y = u - X[X < u]
                        
                        # fit pareto on exedances 
                        # shape, location and scale
                        xi, _, sigma = genpareto.fit(Y, floc = 0)
                        
                        para[i] = [u, xi, sigma]
                        
                except:
                    para[i] = np.repeat(np.nan,n_para[model])

                pbar.update(1)
        
        return(para)
    
    @staticmethod 
    def _get_init_para_cov(cov):
        """
        Predetermined initial parameters for different choises of covariance
        functions
        """
        
        if cov in ('Gudmundson', 'exp'):
            para_init = (10, 10) # l , variance
        elif cov == 'Matern':
            para_init = (50,2.5, 1) # l, nu, variance 
        
        return(para_init)
    
    @staticmethod
    def _unpack_cov_para(cov,cov_para):
        """
        Unpacks parameters and put into dictonary dependin on the covariance 
        model 
        """
        if cov in ('Gudmundson', 'exp'):
            cov_dict = {'l': cov_para[0], 'var': cov_para[1]}
        elif cov == 'Matern':
            cov_dict = {'l': cov_para[0], 
                        'nu': cov_para[1], 
                        'var': cov_para[2]}
        return(cov_dict)
    
    
# =============================================================================
# Some utility functions not included in the class 
# =============================================================================

def factors(n):    
    factors = reduce(list.__add__, ([i, n//i] \
                    for i in range(1, int(n**0.5) + 1) if n % i == 0))
    factors.sort()
    return(factors)


def quantile_est(X, zeta, axis = 0):
    """
    Quantile estimate for X (based on interpolation since no need to pick specific order statistic)
    """
    return(np.quantile(X,zeta, axis = axis, method ='higher'))
 

def find_nearest_idx(array,value):
    """
    Find index in array closest to the given value
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx-1
    else:
        return idx
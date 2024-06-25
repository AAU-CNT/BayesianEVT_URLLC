# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:01:08 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0,'../libraries')
from evt import mean_residual_life
import h5py
data_path = 'generate_quadriga/'
from scipy.stats import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.ticker import AutoMinorLocator

data_index = 0
N_test = 100

f_raw = h5py.File(data_path + f'Stored/Distribution_map_{data_index}_radio_map.h5','r')
D, N = f_raw['fading_samples'].shape

p = []
idx_all = np.arange(0,D, D/N_test).astype('int')

count = 8
while count < N_test:
    try:

        idx = idx_all[count]
        # idx = 1760
        
        snr = f_raw['fading_samples'][idx, :N//2]*1e12
        X = np.log2(1 + snr)
        
        # =============================================================================
        #%% Find u via mean residual life plot
        # =============================================================================
        u_min = np.quantile(X,1e-4)
        # u_min = min(X)
        u_max = np.quantile(X,1e-1)
        mrl = mean_residual_life(X,u_min = u_min, u_max = u_max, log_u = False)
        
        fig, ax = plt.subplots()
        ax.plot(mrl[:,0], mrl[:,1])
        ax.fill_between(mrl[:,0],
                          mrl[:,1] - 1.96*mrl[:,2],
                          mrl[:,1] + 1.96*mrl[:,2],
                          alpha = 0.5,
                          color = 'gray')
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    
        
        ax.grid(which = 'minor')
        ax.grid(which = 'major', color = 'k')
        plt.show()
        
        u = eval(input('Enter threshold : '))
        # =============================================================================
        #%% fit 
        # =============================================================================
        
        Y = u - X[X < u]
        
        # fit pareto
        
        xi_fit, _, scale_fit = genpareto.fit(Y, floc = 0) # shape, location and scale
        print(f'xi : {xi_fit:.2f}, sigma : {scale_fit:.2f}')
        
        
        # =============================================================================
        #%% Get probabilites for CDF
        # =============================================================================
        
        r = np.linspace(X.min(),u, 1000)
        
        # fitted pareto
        P_below_u_emp = np.mean(X < u)
        P_pareto_cdf_fit = ((1 + xi_fit*(u-r)/scale_fit)**(-1/xi_fit))*P_below_u_emp
        
        # emperical CDF
        x_rng = np.linspace(X.min(), np.quantile(X,0.3), 1000)
        # x_rng = np.linspace(X.min(), X.max(), 1000)
        F = ECDF(X)
        
        print(f'eps_u = {P_below_u_emp:.2e}')
        
        # =============================================================================
        #%% plot
        # =============================================================================
        
        #CDF
        plt.semilogy(x_rng, F(x_rng),'o-', label = 'Emperical CDF',
                     markevery = 30)
        plt.semilogy(r, P_pareto_cdf_fit,'-', label = 'Fitted Pareto')
        plt.xlabel('Capacity [bits/s/Hz]')
        plt.ylim(1e-6,1)
        plt.axvline(u, label = 'EVT Threshold', c = 'C2')
        plt.legend()
        plt.title('Fitted CDF using threshold model from EVT')
        plt.show()
    
        ans = input('Accept threshold (y/n): ')
        if ans == 'exit':
            break
        if ans == 'y':
            count +=1 
            p.append(P_below_u_emp)
    except:
        pass 
        
np.save(f'result_files/p_threshold_manual_idx_{data_index}.npy', p)
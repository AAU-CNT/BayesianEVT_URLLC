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
from scipy.stats import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
import sys
from stat_radio_map import stat_radio_map
path = '' # insert path to data

SNR_scale = 1e10

mod_all = stat_radio_map(5,path, data_origin = 'APMS',
                     SNR_scale = SNR_scale)
C_all = np.log2(1 + mod_all.fading*mod_all.SNR_scale)

#%%
u_all = []

N_test = 127
count = 0
while count <= N_test - 1:
    try:
        X = C_all[count]
        
        # =============================================================================
        #%% Find u via mean residual life plot
        # =============================================================================
        # u_min = np.quantile(X,1e-4)
        u_min = min(X)
        u_max = np.quantile(X,0.5)
        # u_max = max(X)
        mrl = mean_residual_life(X,u_min = u_min, u_max = u_max, log_u = False)
        
        fig, ax = plt.subplots()
        ax.plot(mrl[:,0], mrl[:,1])
        ax.fill_between(mrl[:,0],
                          mrl[:,1] - 1.96*mrl[:,2],
                          mrl[:,1] + 1.96*mrl[:,2],
                          alpha = 0.5,
                          color = 'gray')
        ax.set_title(f'Location {count}')
        ax.ticklabel_format(style = 'sci', axis = 'x', scilimits=(0,0))
    
        
        ax.grid(which = 'minor')
        ax.grid(which = 'major', color = 'k')
        plt.show()
        
        try:
            u = eval(input('Enter threshold : '))
        except:
            break
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
        
        r = np.linspace(1.1*X.min(),u, 1000)
        
        # fitted pareto
        P_below_u_emp = np.mean(X < u)
        P_pareto_cdf_fit = ((1 + xi_fit*(u-r)/scale_fit)**(-1/xi_fit))*P_below_u_emp
        
        # emperical CDF
        x_rng = np.linspace(X.min(), u, 1000)
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
        if ans == 'y':
            count +=1 
            u_all.append(u)
        elif ans == 'exit':
            break
    except:
        pass 
        
np.save('result_filesthreshold_manual_apms.npy', u_all)

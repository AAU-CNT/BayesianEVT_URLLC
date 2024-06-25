# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:05:30 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0,'../libraries')
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from statsmodels.distributions.empirical_distribution import ECDF
from rate_select_non_para import find_l    
import plotsetup

version = '0'
with open(f'result_files/rate_select_v{version}.pickle','rb') as f:
    dat = pickle.load(f)

methods = list(dat['p_out'].keys())
n_rng  = dat['n_rng']
eps = dat['eps']
m = 10000
N_fade = 4001
alpha = 0.05
N_CDF = 1000
m = 10_000
rng = 10**(np.linspace(-4,-1,N_CDF))
N_eval = len(dat['idx_eval'])
N_n = len(n_rng)
markers = ['^','s','v','o']
names = ['Baseline non-parametric', 'Bayesian non-parametric', 'Baseline EVT', 'Bayesian EVT']
idx_show = [0,1,2,5,6,7]
compute_p_out_uncertainty = True

F_rng = np.zeros((4, N_n, N_CDF))
p_meta_stats = np.zeros((4, N_n,3)) # mean, lower, upper
method_order = [1,3,0,2]

p_meta_non_para_bay_theo = (1 - np.array([find_l(eps,dat['delta'], n)[1] for n in n_rng]))*100
p_meta_non_para_bay_sim = np.mean(dat['p_out']['freq_nonpara'] < eps,axis = 0)*100
print(f'Maximum difference between theoerical and analtycial: {np.abs(p_meta_non_para_bay_theo- p_meta_non_para_bay_sim).max():.2f}%')

# =============================================================================
# %% Plot selected outage curves
# =============================================================================

idx_show = [0,1,2,4,6, 7]
# idx_show = range(len(n_rng))
c = ['C0','C1','C2','C6']


fig, ax = plt.subplots(nrows = len(idx_show), figsize = (8,14))
fig.subplots_adjust(hspace = 0.3)
for i, idx in enumerate(idx_show):
    
    n = n_rng[idx]

    for j in range(len(methods)):
        j_reorder = method_order[j]
        ax[i].plot(rng,ECDF(dat['p_out'][methods[j_reorder]][:,idx])(rng),
                   label = names[j_reorder], 
                   marker = markers[j_reorder], 
                   markevery = 100,
                   c = c[j])
    ax[i].set_title(f'n : {n}')
    ax[i].axhline(1-dat['delta'], c = 'k', label = r'Target confidence $1 - \delta$',
                  linestyle = '--')
    ax[i].axvline(dat['eps'], c = 'r', label = r'Target PEP $\epsilon$',
                  linestyle = '-.')
    ax[i].set_ylabel(r'ECDF')
    ax[i].set_xscale('log')
    ax[i].set_xlim(1e-3,2e-2)
    ax[i].set_ylim(-0.05,1.05)
    
    if i != len(idx_show) - 1:
        ax[i].set_xticks([])

    else:
        ax[i].set_xlabel(r'Outage probability $p_{out}$')
        ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.45),
          fancybox=True, shadow=True, ncol=2)

ax[-1].set_xticks([1e-3,3.162e-3,1e-2])
ax[-1].set_xticklabels([r'$10^{-3}$',r'$3\cdot 10^{-3}$', r'$10^{-2}$'])
fig.savefig('../plots/fig7.pdf',bbox_inches = 'tight')


# =============================================================================
#%% plot throughput as boxplot
# =============================================================================

fig, ax = plt.subplots(figsize = (8,4))
ax.axhline(1, c = 'k')
ax.axhspan(1,2, color = 'gray', alpha = 0.5)
for i in range(len(methods)):
    i_reorder = method_order[i]
    method = methods[i_reorder]
    
    Q1 = np.quantile(dat['omega'][method],0.25, axis = 0)
    Q3 = np.quantile(dat['omega'][method],0.75, axis = 0)
    median = np.median(dat['omega'][method], axis = 0)
    ax.fill_between(n_rng, Q1, Q3, alpha = 0.3, 
                    color = c[i])
    ax.plot(n_rng,median, label = names[i_reorder], marker = markers[i_reorder], c = c[i])
        


ax.legend(loc = 'lower right', bbox_to_anchor=(1,0.05))
ax.set_xlim(-20, 5000)
ax.set_ylim(-0.05,1.15)
ax.set_xscale('symlog', linthresh = 100)
ax.set_xlabel('\#Observations n')
ax.grid()

ax.set_ylabel(r'Normalized Throughput $\tilde{R}_{\epsilon}$')
fig.savefig('../plots/fig8.pdf', bbox_inches = 'tight')



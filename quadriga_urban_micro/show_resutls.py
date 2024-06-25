# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:05:30 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
from statsmodels.distributions.empirical_distribution import ECDF
import sys
sys.path.insert(0,'../libraries')
import plotsetup
from rate_select_non_para import find_l  


version = '0'
data_index = 8
with open(f'result_files/rate_select_{data_index}_v{version}.pickle','rb') as f:
    dat = pickle.load(f)
    

p_out_all = [dat['p_out'][key] for key in dat['p_out']]
    

methods = list(dat['p_out'].keys())
# methods = ['freq_nonpara']
n_rng  = dat['n_rng']
N_n = len(n_rng)
N_fade = int(1e6)
m = 10000
SNR_scale = 1e12
alpha = 0.05
N_CDF = 500
eps = dat['eps']
N_eval = dat['R_all']['freq_nonpara'].shape[0]
markers = ['^','s','v','o']
names = ['Baseline non-parametric', 'Bayesian non-parametric', 'Baseline EVT', 'Bayesian EVT']
compute_p_out_uncertainty = False
method_order = [1,3,0,2]

rng = 10**(np.linspace(-6,-3,N_CDF))

def to_scientific(n):
    if n == 0:
        return(r'$0$')
    exp = int(np.log10(n))
    man = round(n/10**exp,1)
    if man == 1:
        n_sci = r'$10^{%d}$' % exp
    else:
        n_sci = r'$%.1f\cdot 10^{%d}$' %(man,exp)
    return(n_sci)


p_meta_non_para_bay_theo = (1 - np.array([find_l(eps,dat['delta'], n)[1] for n in n_rng]))*100
p_meta_non_para_bay_sim = np.mean(dat['p_out']['freq_nonpara'] < eps,axis = 0)*100
print(f'Maximum difference between theoerical and analtycial: {np.abs(p_meta_non_para_bay_theo- p_meta_non_para_bay_sim).max():.2f}%')

# =============================================================================
# %% Plot selected outage curves
# =============================================================================

idx_show = [0,1,2,6,9]
# idx_show = range(len(n_rng))
c = ['C0','C1','C2','C6']


fig, ax = plt.subplots(nrows = len(idx_show), figsize = (8,12)) # 6, 10
fig.subplots_adjust(hspace = 0.3)
for i, idx in enumerate(idx_show):
    
    n = n_rng[idx]

    for j in range(len(methods)):
        j_reorder = method_order[j]
        ax[i].plot(rng,ECDF(dat['p_out'][methods[j_reorder]][:,idx])(rng), 
                   label = names[j_reorder], 
                   marker = markers[j_reorder], 
                   markevery = 50,
                   c = c[j])
        
    # ax[i].set_title(f'n : {to_scientific(n)}')
    if n < 1e5:
        ax[i].set_title(f'n : {n}')
    else:
        ax[i].set_title(f'n : {n} = {to_scientific(n)}')
    
    ax[i].axhline(1-dat['delta'], c = 'k', label = r'Target confidence $1 - \delta$',
                  linestyle = '--')
    ax[i].axvline(dat['eps'], c = 'r', label = r'Target PEP $\epsilon$',
                  linestyle = '-.')
    ax[i].set_ylabel(r'ECDF')
    ax[i].set_xscale('log')
    ax[i].set_xlim(1e-5,2e-4)
    ax[i].set_ylim(-0.05,1.05)
    
    if i != len(idx_show) - 1:
        ax[i].set_xticks([])

    else:
        ax[i].set_xlabel(r'Outage probability $p_{out}$')
        ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.45),
          fancybox=True, shadow=True, ncol=2)


ax[-1].set_xticks([1e-5,3.162e-5,1e-4])
ax[-1].set_xticklabels([r'$10^{-5}$',r'$3\cdot 10^{-5}$', r'$10^{-4}$'])

fig.savefig('../plots/fig2.pdf',bbox_inches = 'tight')

# =============================================================================
# %% plot meta probability
# =============================================================================

fig, ax = plt.subplots(figsize = (8,4))

for i in range(len(methods)):
    i_reorder = method_order[i]

    p_meta = np.mean(dat['p_out'][methods[method_order[i]]] < eps,axis = 0)*100
    ax.plot(n_rng, p_meta, label = names[i_reorder], marker = markers[i_reorder],
            c = c[i])

ax.set_xlabel('\#Observations n')
ax.set_ylabel(r'Meta probability $\tilde{p}_{\epsilon}$ [\%]')
ax.axhline((1 - dat['delta'])*100, c = 'k', label = r'Target $1- \delta$',
            linestyle = '--')    
ax.legend()
# ax.plot(n_rng, p_meta_non_para_bay_theo,'C0--')
ax.set_xscale('symlog', linthresh = 100)
ax.set_xlim(-10,1.5e6)
ax.set_ylim(50,105)
ax.grid()
fig.savefig('../plots/fig3.pdf', bbox_inches = 'tight')


# =============================================================================
#%% plot throughput as boxplot
# =============================================================================

fig, ax = plt.subplots(figsize = (8,4))
ax.axhline(1, c = 'k')
ax.axhspan(1,2, color = 'gray', alpha = 0.3)
for i in range(len(methods)):
    i_reorder = method_order[i]
    method = methods[i_reorder]
    
    Q1 = np.quantile(dat['omega'][method],0.25, axis = 0)
    Q3 = np.quantile(dat['omega'][method],0.75, axis = 0)
    median = np.median(dat['omega'][method], axis = 0)
    ax.fill_between(n_rng, Q1, Q3, alpha = 0.3, 
                    color = c[i])
    ax.plot(n_rng,median, label = names[i_reorder], marker = markers[i_reorder], c = c[i])
    # ax.plot(n_rng,mean + std, c = c[i], linestyle = '--')
    # ax.plot(n_rng,mean - std, c = c[i], linestyle = '--')
        


ax.legend(loc = 'lower right', bbox_to_anchor=(1,0.05))
ax.set_xlim(-20, 1.5e6)
ax.set_ylim(-0.05,1.1)
ax.set_xscale('symlog', linthresh = 100)
ax.set_xlabel('\#Observations n')
ax.grid()

ax.set_ylabel(r'Normalized Throughput $\tilde{R}_{\epsilon}$')
fig.savefig('../plots/fig4.pdf', bbox_inches = 'tight')


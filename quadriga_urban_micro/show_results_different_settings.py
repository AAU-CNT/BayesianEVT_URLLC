# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:05:30 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import sys
sys.path.insert(0,'../libraries')
import plotsetup
import matplotlib.text as mtext


with open('result_files/rate_select_8_v0.pickle','rb') as f:
    dat = pickle.load(f)

with open('result_files/rate_select_8_v_low_d.pickle','rb') as f:
    dat_low_d = pickle.load(f)

with open('result_files/rate_select_8_v_low_m.pickle','rb') as f:
    dat_low_m = pickle.load(f)


    
methods = list(dat['p_out'].keys())
# methods = ['freq_nonpara']
n_rng  = dat['n_rng']
N_n = len(n_rng)
alpha = 0.05
eps = dat['eps']
markers = ['^','s','v','o']
c = ['C0','C1','C2','C6']
names = ['$m = 10^6, d = 500$', '$m = 10^6, d = 500$', 'Non-parametric','EVT']
names_low_d = [r'$m = 10^6, d = 100$', '$m = 10^6, d = 100$']
names_low_m = [r'$m = 10^4, d = 500$', '$m = 10^4, d = 500$']
method_order = [1,3,0,2]


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

# =============================================================================
# %% plot meta probability and throughput together
# =============================================================================

fig, ax = plt.subplots(figsize = (8,6), nrows = 2)


# meta-probability
for i in range(4):
    i_reorder = method_order[i]

    p_meta = np.mean(dat['p_out'][methods[method_order[i]]] < eps,axis = 0)*100
    ax[0].plot(n_rng, p_meta, marker = markers[i_reorder],
            c = c[i])
    
    if i < 2:
        p_meta = np.mean(dat_low_d['p_out'][methods[method_order[i]]] < eps,axis = 0)*100
        ax[0].plot(n_rng, p_meta, marker = markers[i_reorder],
                c = c[i],
                linestyle = '--') 
        p_meta = np.mean(dat_low_m['p_out'][methods[method_order[i]]] < eps,axis = 0)*100
        ax[0].plot(n_rng, p_meta, marker = markers[i_reorder],
                c = c[i],
                linestyle = ':') 
        


ax[0].set_title(r'Meta probability')
ax[0].axhline((1 - dat['delta'])*100, c = 'k', label = r'Target $1- \delta$',
            linestyle = '--')    
ax[0].set_xscale('symlog', linthresh = 100)
ax[0].set_xlim(-10,1.5e6)
# ax[0].set_ylim(85,101)
ax[0].grid()
ax[0].legend(loc = 'lower right')
ax[0].set_ylabel(r'$\tilde{p}_{\epsilon}$ [\%]')
ax[0].set_xticklabels([])


# throughput
ax[1].axhline(1, c = 'k')
ax[1].axhspan(1,2, color = 'gray', alpha = 0.5)
for i in range(4):
    i_reorder = method_order[i]
    method = methods[i_reorder] 
    median = np.median(dat['omega'][method], axis = 0)
    ax[1].plot(n_rng,median, label = names[i], marker = markers[i_reorder], c = c[i])
    
    if i < 2:
        median = np.median(dat_low_d['omega'][method], axis = 0)
        ax[1].plot(n_rng,median, label = names_low_d[i], marker = markers[i_reorder], c = c[i],
                   linestyle = '--')
        median = np.median(dat_low_m['omega'][method], axis = 0)
        ax[1].plot(n_rng,median, label = names_low_m[i], marker = markers[i_reorder], c = c[i],
                   linestyle = ':')


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title
    
handles, labels = ax[1].get_legend_handles_labels()
handles.insert(0, 'Bayesian Non-parametric')
labels.insert(0, '')
handles.insert(4, 'Baeysian EVT')
labels.insert(4, '')
handles.insert(8, 'Baselines')
labels.insert(8, '')
ax[1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.3),
             fancybox=True, shadow=True, ncol=3, handlelength=3,
             handler_map={str: LegendTitle({'fontsize': 12})})
# ax[1].legend(loc = 'lower right', bbox_to_anchor=(1,0.05), handlelength=3)
ax[1].set_xlim(-20, 1.5e6)
ax[1].set_ylim(.6,1.08)
ax[1].set_xscale('symlog', linthresh = 100)
ax[1].set_xlabel('\#Observations n')
ax[1].grid()
ax[1].set_ylabel(r'$\tilde{R}_{\epsilon}$')
ax[1].set_title(r'Normalized Throughput (median)')
fig.savefig('../plots/fig5.pdf', bbox_inches = 'tight')

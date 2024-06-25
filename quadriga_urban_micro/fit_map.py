# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:29:02 2023

@author: Tobias Kallehauge
"""

import sys 
sys.path.insert(0,'../libraries')
data_path = '../generate_quadriga'
from stat_radio_map import stat_radio_map
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import plotsetup

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


data_index = 0
np.random.seed(72)

# reliability parameters
eps = 1e-4
delta = 0.05 # 5%

# model parameters
SNR_scale = 1e12
variable = 'capacity'

# parameters for radio map
cov = ['Matern', 'Matern', 'Matern'] # q, ln_f_q, xi
log_q = True

# parameters for GPD
zeta = 1e-2 # previously 1e-3
N_min = 50 # minium number of samples used in the tail
d = 500
d_test = 200
N_fade_all = int(2e6) # number of fading observations
# split test/trn in 2
trn_slice = slice(N_fade_all//2) 
tst_slice = slice(N_fade_all//2,N_fade_all) 


# number of samples avaialble for rate section
n_rng = np.hstack([0, 10**(np.linspace(3,6,7))]).astype('int')
N_n = len(n_rng)

# load all data 
mod_all = stat_radio_map(data_index,data_path, SNR_scale = SNR_scale)
mod = stat_radio_map(data_index,data_path, SNR_scale = SNR_scale,
                  rx_subsample_mode = 'random_non_uniform', 
                  rx_subsample= d + d_test)
s_trn = mod.s[:d]
s_tst = mod.s[d:]

mod_trn = stat_radio_map(data_index,data_path, SNR_scale = SNR_scale,
                          rx_subsample_mode = 'list', 
                          rx_subsample = s_trn)  


idx = []
for s in mod_trn.s:
    idx.append(np.linalg.norm(s - mod_all.s,axis = 1).argmin())


# =============================================================================
# %% fit inital map to get prior paramters and initial values for Gibbs sampler
# =============================================================================
# fit GDP to entire map, i.e, [u, xi, sigma]
para_all_GPD = mod_all.fit_map(variable = variable,
                                model = 'GPD',
                                zeta = zeta)
# fit q to map
para_all_q = mod_all.fit_map(variable = variable,
                              model = 'emp_pdf',
                              eps = eps,
                              log = log_q)



# para_all_GPD contains (u, xi, sigma)
# para_all_q contains (q, ln_f_q)
para_all = np.hstack((para_all_q,para_all_GPD[:,1:2])) # (q, ln_f_q, xi)

para = para_all[idx]

# =============================================================================
# interpolate map
# =============================================================================

pred_mean, pred_var, hyper = mod_trn.interpolate_map(mod_all.s, para, cov)


# =============================================================================
#%% Plot 
# =============================================================================

titles = [r'$\epsilon$-outage capacity $C_{\epsilon}$', r'pdf-value at $\epsilon$-quantile $f_Y(Y_{\epsilon})$',
          r'GPD shape parameter $\xi$']
cbar_titles = [r'$C_{\epsilon}$', r'$f_Y(Y_{\epsilon})$',r'$\xi$']

X_true = [np.exp(para_all[:,0]), # quantile
          np.exp(para_all[:,1]), # pdf
          para_all[:,2]] # xi

X_pred = [np.exp(pred_mean[:,0] - pred_var[:,0]), # mode of lognorm
          np.exp(pred_mean[:,1] - pred_var[:,1]), # mode of lognorm
          pred_mean[:,2]]

norms = [Normalize(vmin = X_true[0].min(), vmax= X_true[0].max()),
          LogNorm(vmin = X_true[1].min(), vmax= X_true[1].max()),
          Normalize(vmin = X_true[2].min(), vmax = X_true[2].max())
          ]

for i in range(3):
    # quantile
    norm = norms[i]
    fig, ax = plt.subplots(ncols = 2, figsize = (8,4.5))
    fig.subplots_adjust(wspace = 0.12)
    im = ax[0].imshow(X_true[i].reshape(mod_all.N_rx_side,mod_all.N_rx_side),
                        origin = 'lower',  extent = mod_all.extent, cmap = 'jet', 
                        norm = norm)
    ax[1].scatter(mod_trn.s[:,0],mod_trn.s[:,1],c = 'w', edgecolor = 'k', s = 10,
                  label = 'Training')
    # ax[1].scatter(mod_tst.s[:,0],mod_tst.s[:,1],c = 'k', edgecolor = 'k', s = 10,
    #               label = 'Test')
    ax[1].set_yticks([])
    ax[0].plot([mod_trn.s_BS[0]],[mod_trn.s_BS[0]],markersize = 10,
            marker = 'X', markerfacecolor= 'w', 
            markeredgecolor='k', markeredgewidth=1.0, linewidth = 0,
            label = 'BS',
            clip_on = False, zorder = 4)
    ax[1].plot([mod_trn.s_BS[0]],[mod_trn.s_BS[0]],markersize = 10,
            marker = 'X', markerfacecolor= 'w', 
            markeredgecolor='k', markeredgewidth=1.0, linewidth = 0,
            clip_on = False, zorder = 4)
    ax[1].imshow(X_pred[i].reshape(mod_all.N_rx_side,mod_all.N_rx_side),
                  origin = 'lower', extent = mod_trn.extent, cmap = 'jet', 
                  norm = norm)
    cax = fig.add_axes([ax[1].get_position().x1+0.02,ax[1].get_position().y0,0.03,ax[1].get_position().height])
    cbar = fig.colorbar(im, cax = cax)
    ax[0].set_xlabel('1st coordinate [m]')
    ax[1].set_xlabel('1st coordinate [m]')
    ax[0].set_ylabel('2nd coordinate [m]')
    # cbar.set_label(cbar_titles[i])
    ax[0].legend(loc = 'upper left')
    # ax[1].legend(loc = 'upper left', fontsize = 14)
    fig.suptitle(titles[i], y = .95)
    ax[0].set_title('Reference')
    ax[1].set_title(f'Interpolated map from {mod_trn.N_rx} locations') 
    fig.savefig(f'../plots/fig1_{i}.pdf', bbox_inches = 'tight')
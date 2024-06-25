# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:01:27 2024

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import sys
sys.path.insert(0,'../libraries')
from stat_radio_map import stat_radio_map
from scipy.interpolate import griddata
path = ''
import warnings
warnings.filterwarnings('ignore')
np.random.seed(72)

SNR_scale = 1e10
variable = 'capacity'
# parameters for radio map
cov = ['Gudmundson', 'Matern', 'Matern'] 
log_q = True
eps = 1e-2
rx_subsample = 50
rx = (88.57397, -0.90699, 1.15) 
extent = [-.5, 112, -15, 53.1]
zeta = 0.4
N_trn = 4000
idx_fade_all = np.arange(8001)
np.random.shuffle(idx_fade_all)
idx_fade_trn = idx_fade_all[:N_trn]
idx_fade_tst = idx_fade_all[N_trn:]


mod_all = stat_radio_map(5,path, data_origin = 'APMS',
                     SNR_scale = SNR_scale,
                     fade_subsample = idx_fade_tst)
mod = stat_radio_map(5,path, data_origin = 'APMS',
                     SNR_scale = SNR_scale,
                     rx_subsample_mode='random',
                     rx_subsample= rx_subsample,
                     fade_subsample = idx_fade_trn)

idx = []
for s in mod.s:
    idx.append(np.linalg.norm(s - mod_all.s,axis = 1).argmin())


# =============================================================================
#%% Interpolate map
# =============================================================================
# generate uniform grid for GP
N_grid = 100
s0 = np.linspace(extent[0], extent[1], N_grid)
s1 = np.linspace(extent[2], extent[3], N_grid)
s3 = np.repeat(1.15, N_grid**2)
SS0, SS1 = np.meshgrid(s0,s1)
SS = np.stack((SS0.flatten(), SS1.flatten(),s3), axis = 1)


para_all_GPD = mod_all.fit_map(variable = variable,
                                model = 'GPD',
                                zeta = zeta)
# fit q to map
para_all_q = mod_all.fit_map(variable = variable,
                        model = 'emp_pdf',
                        eps = eps,
                        log = log_q)

para_GPD = mod.fit_map(variable = variable,
                                model = 'GPD',
                                zeta = zeta)
# fit q to map
paral_q = mod.fit_map(variable = variable,
                        model = 'emp_pdf',
                        eps = eps,
                        log = log_q)

para_all = np.hstack((para_all_q,para_all_GPD[:,1:2])) # (q, ln_f_q, xi)
para = np.hstack((paral_q,para_GPD[:,1:2]))


# =============================================================================
#%% interpolate map
# =============================================================================
pred_mean, pred_var, hyper = mod.interpolate_map(SS, para, cov)

# just at known locations
pred_mean_eval, pred_var_eval, hyper_eval = mod.interpolate_map(mod_all.s, para, cov)


# =============================================================================
#%% Setup some plotting
# =============================================================================

n = 100
XX, YY = np.meshgrid(np.linspace(extent[0], extent[1],n),
                     np.linspace(extent[2], extent[3],n))


buildings = ([[[52, 53],
            [50.6+1, 28.4-1.5],
            [47.6+1,28.4-1.5],
            [47.6+1,25-1.5],
            [50.6+1,25-1.5],
            [51.4, 13.9],
            [38.8, 13.9],
            [38.6,  2.4],
            [-.3 ,  3. ],
            [-0.2, 53]],
            [[ 59.6,  53.],
            [ 59.5,  37],
            [83, 37],
            [83, 40.8],
            [ 87.6,  40.8],
            [ 87.6,  27],
            [ 99,  27],
            [ 98.9,  14],
            [ 86.5,  14. ],
            [ 86.5,   1.4],
            [111.9,   1.1],
            [111.9,  53]],
            [[ -.3,  -6.3],
            [ -0.3, -20.2],
            [111.8, -20.5],
            [111.8,  -7.3],
            [ 71.7,  -7. ],
            [ 71.7, -12.8],
            [ 52. , -12.6],
            [ 52. ,  -7.1]]])


def to_path(b):
    x = [c[0] for c in b]
    y = [c[1] for c in b]
    return(x,y)

# =============================================================================
#%% plot
# =============================================================================

X_true = [np.exp(para_all[:,0]), # quantile
          np.exp(para_all[:,1]), # pdf
          para_all[:,2]] # xi

X_pred = [np.exp(pred_mean[:,0] - pred_var[:,0]), # mode of lognorm
          np.exp(pred_mean[:,1] - pred_var[:,1]), # mode of lognorm
          pred_mean[:,2]]

norms = [LogNorm(vmin = X_true[0].min(), vmax= X_true[0].max()),
         LogNorm(vmin = X_true[1].min(), vmax= X_true[1].max()),
         Normalize(vmin = X_true[2].min(), vmax = X_true[2].max())
         ]


titles = ['$\epsilon$-outage capacity $C_{\epsilon}$', r'pdf-value at $\epsilon$-quantile $f_Y(Y_{\epsilon})$',
          r'GPD shape parameter $\xi$']

for i , (pred, true , title, norm) in enumerate(zip(X_pred, X_true, titles, norms)):


    # Interpolate true values based on closest observations    
    ZZ_true = griddata((mod_all.s[:,0], mod_all.s[:,1]), true, (XX, YY), method='nearest')

    fig, ax = plt.subplots(ncols = 2, figsize = (8,4.5))
    fig.subplots_adjust(wspace = 0.05)
    im = ax[0].imshow(ZZ_true, extent=extent, origin='lower', aspect='auto', cmap='jet', norm = norm)
    # ax[0].scatter(mod_all.s[:,0], mod_all.s[:,1], s = 5, c = 'w', edgecolor = 'k', label = 'TX', linewidth = 0.5)
    ax[0].plot([rx[0]],[rx[1]],markersize = 10,
            marker = 'X', markerfacecolor= 'w', 
            markeredgecolor='k', markeredgewidth=1.0, linewidth = 0,
            label = 'BS')
    
    ax[0].fill(*to_path(buildings[0]), c ='w', edgecolor = 'k', hatch = '/', label = 'Buildings')
    ax[0].fill(*to_path(buildings[1]), c ='w', edgecolor = 'k', hatch = '/')
    ax[0].fill(*to_path(buildings[2]), c ='w', edgecolor = 'k', hatch = '/')
    
    ax[1].imshow(pred.reshape(N_grid,N_grid), extent=extent, origin='lower', aspect='auto', cmap='jet', norm = norm)
    ax[1].scatter(mod.s[:,0], mod.s[:,1], s = 10, c = 'w', edgecolor = 'k', label = 'Observed\nTX', linewidth = 0.5)
    ax[1].plot([rx[0]],[rx[1]],markersize = 10,
            marker = 'X', markerfacecolor= 'w', 
            markeredgecolor='k', markeredgewidth=1.0, linewidth = 0,
            label = 'BS')
    ax[1].fill(*to_path(buildings[0]), c ='w', edgecolor = 'k', hatch = '/', label = 'Buildings')
    ax[1].fill(*to_path(buildings[1]), c ='w', edgecolor = 'k', hatch = '/')
    ax[1].fill(*to_path(buildings[2]), c ='w', edgecolor = 'k', hatch = '/')
    ax[1].set_yticks([])

    
    # labels
    cax = fig.add_axes([ax[1].get_position().x1+0.02,ax[1].get_position().y0,0.035,ax[1].get_position().height])
    cbar = fig.colorbar(im, cax = cax, cmap = 'jet', norm = norm)
    
    ax[0].set_xlim(-5,115)
    ax[0].set_ylim(-23,55)
    ax[1].set_xlim(-5,115)
    ax[1].set_ylim(-23,55)
    ax[0].set_xlabel('1st coordinate [m]')
    ax[1].set_xlabel('1st coordinate [m]')
    ax[0].set_ylabel('2nd coordinate [m]')

    legend = ax[0].legend(fontsize = 12, loc = 'upper left')
    legend.get_frame().set_alpha(.95)
    
    # titles
    ax[0].set_title('Reference')
    ax[1].set_title('Interpolated map from 50 locations')
    fig.suptitle(title, y = 1.01)
    fig.savefig(f'../plots/fig6_{i}.pdf', bbox_inches = 'tight')


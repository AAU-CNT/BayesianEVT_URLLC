# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:06:08 2023

@author: Tobias Kallehauge
"""


import numpy as np
import pandas as pd
import h5py


D = 127 # number of ppints (excluding blocked points)
eps = 0.01
f_low = 2.000 # GHZ
f_high = 10.000 # GHZ

meta = pd.read_csv('data/meta_data.csv', index_col = 0, nrows = D)
meta.index.name = 'Nr.'

# load single datafile to count the number of frequency points inrange
dat = pd.read_csv(f'/data/{0}.txt', header = 16, sep = '\t',
                  index_col = 0, 
                  usecols = ['PNT', 'FREQ1.GHZ', 'LOGMAG1','PHASE2.DEG'])
N = sum((dat['FREQ1.GHZ'] >= f_low) & (dat['FREQ1.GHZ'] <= f_high))

# =============================================================================
# Initialize h5 file 
# =============================================================================

f_pro = h5py.File('data/processed_map.h5', 'w')

# initialize datasets
f_pro.create_dataset('mean', shape = (D,), dtype = 'f')
f_pro.create_dataset('fading', shape = (D,N), dtype = 'f')
f_pro.create_dataset('quantiles', shape = (D,), dtype = 'f')
f_pro.create_dataset('coordinates', data = meta[['x','y','z']].values)
f_pro.create_dataset('index', data = meta.index.values)
f_pro.create_dataset('rx', data = np.array([88.57397, -0.90699, 1.15])) # with 26 as reference
f_pro.create_dataset('f_rng', data = np.array([f_low,f_high])) # with 26 as reference
f_pro.create_dataset('eps', data = eps) # with 26 as reference


# H = np.zeros(len(meta), dtype = 'complex')
# P_nb_mean_db = np.zeros(len(meta))
# P_nb_q_db = np.zeros(len(meta))

for i, nr in enumerate(meta.index):    
    dat = pd.read_csv(f'data/{nr}.txt', header = 16, sep = '\t',
                      index_col = 0, 
                      usecols = ['PNT', 'FREQ1.GHZ', 'LOGMAG1','PHASE2.DEG'])
    dat.columns = ['FREQ.GHZ', 'LOGMAG', 'PHASE.DEG']
    dat = dat[(dat['FREQ.GHZ'] >= f_low) & (dat['FREQ.GHZ'] <= f_high)]
    
    # compute narrowband power
    # first frequyency response
    dat['H'] = 10**(dat['LOGMAG']/20)*np.exp(1j*dat['PHASE.DEG']*np.pi/180)
    # narrowband power is just the same as the frequency response
    h_nb = (np.abs(dat['H'])**2).values
    
    # save values
    f_pro['fading'][i] = h_nb # all fading samples
    f_pro['quantiles'][i] = np.quantile(h_nb, eps)
    f_pro['mean'][i] = np.mean(h_nb)

f_pro.close()

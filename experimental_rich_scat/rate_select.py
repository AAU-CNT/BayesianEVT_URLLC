# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:29:02 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0,'../libraries')
path = ''
from stat_radio_map import stat_radio_map
import numpy as np
import pickle
from rate_select_GPD import rate_select_GPD
from rate_select_non_para import rate_select_non_para
import multiprocessing as mp
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

np.random.seed(72)
version = '0'

# reliability parameters
eps = 1e-2
delta = 0.05 # 5%
rx_subsample = 50

# model parameters
SNR_scale = 1e10
N_fade_all = 8001
N_fade_trn = 4000
variable = 'capacity'
L = 130 
methods = ['freq_nonpara', 'bay_nonpara', 'freq_GPD', 'bay_GPD']

# parameters for radio map
cov = ['Gudmundson', 'Gudmundson', 'Matern'] # q, ln_f_q, xi
log_q = True

# parameters for GPD
zeta = 0.4 # 0.4
N_min = 50 # minium number of samples used in the tail

# gibbs parameters
gibbs_params = {'T': 10_000,  
                'burn_in': 1000,
                'calibrate_proposal' : True,
                'verbose': False}

# number of samples avaialble for rate section
n_rng = np.hstack([0, 10**(np.linspace(np.log10(51),np.log10(N_fade_trn),7))]).astype('int')
N_n = len(n_rng)

# multiprocessing parameters
N_workers = 40 

def get_R(C_all, pred_mean, pred_var):
    
    R = {method : np.zeros(N_n) for method in methods}    
    
    # setup prior
    prior = {'prior_q' : (pred_mean[0], pred_var[0]), # mean, var for q (in log domain)
             'prior_xi': (pred_mean[2], pred_var[2])} # mean, var for xi
    

    for j, n in enumerate(n_rng):
        try:
            zeta_n = zeta
            if n*zeta_n <= N_min and n > 0: # too high variance in inference so set a higher zeta
                zeta_n = N_min/n
            
            C = np.random.choice(C_all, n, replace = False)
            
            R['freq_nonpara'][j] = rate_select_non_para(eps,
                                                          X = C,
                                                          model = 'capacity',
                                                          method = 'freq',
                                                          target = 'pcr',
                                                          delta = delta)
            
            # ln_f_q is lognormal so get mode of prediction of pdf value
            f_q = np.exp(pred_mean[1] - pred_var[1]) 
            
            prior = {'q_mean': pred_mean[0],
                      'q_var' : pred_var[0],
                      'pdf_eps': f_q}
            
            R['bay_nonpara'][j] = rate_select_non_para(eps, 
                                                        X = C,
                                                        model = 'capacity',
                                                        method = 'bay',
                                                        target = 'pcr',
                                                        prior = prior,
                                                        delta = delta)

            R['freq_GPD'][j] = rate_select_GPD(eps,
                                                zeta_n,
                                                method = 'freq',
                                                C = C,
                                                delta = delta,
                                                freq_test = 'profile')
        
            prior = {'prior_q' : (pred_mean[0], pred_var[0]), # mean, var for q (in log domain)
                      'prior_xi': (pred_mean[2], pred_var[2])} # mean, var for xi
        
            R['bay_GPD'][j]  = rate_select_GPD(eps,
                                                zeta_n, 
                                                method = 'bay',
                                                C = C, 
                                                gibbs_params= gibbs_params,
                                                prior = prior,
                                                delta = delta,
                                                log_q = log_q)
                
        except:
            for method in methods:
                R[method][j] = np.nan

                
    return(R)
        

if __name__ == '__main__':
    R_all_L = {method : [] for method in methods}
    p_out_L = {method : [] for method in methods}
    omega_L = {method : [] for method in methods}
    idx_eval_L = []
    t0 = time.time()
    l = 0
    while l < L:
        if l > 0:
            time_per_loop_avg = (time.time() - t0)/l
            time_left = time_per_loop_avg*(L-l)
            time_left_str = f'Approximate time left: {time_left/60/60:.1f} hours'
            
        else:
            time_left_str = ''
        
        print(f'Selecting rates rerun {l+1} of {L}. {time_left_str}', flush = True)
# =============================================================================
# Setup models for train and test         
# =============================================================================
        idx_fade_all = np.arange(N_fade_all)
        np.random.shuffle(idx_fade_all)
        idx_fade_trn = idx_fade_all[:N_fade_trn]
        idx_fade_tst = idx_fade_all[N_fade_trn:]

        mod_loc_trn = stat_radio_map(5,path, data_origin = 'APMS',
                              SNR_scale = SNR_scale,
                              rx_subsample_mode = 'random',
                              rx_subsample = rx_subsample,
                              fade_subsample= idx_fade_trn)
        mod_all_trn = stat_radio_map(5,path, data_origin = 'APMS',
                              SNR_scale = SNR_scale,
                              fade_subsample = idx_fade_trn)       

        idx = []
        for s in mod_loc_trn.s:
            idx.append(np.linalg.norm(s - mod_all_trn.s,axis = 1).argmin())
        
# =============================================================================
# Fit parameters         
# =============================================================================
        print('Fitting parameters...', end = '', flush = True)
        # fit GDP to entire map, i.e, [u, xi, sigma]
        para_GPD = mod_loc_trn.fit_map(variable = variable,
                               model = 'GPD',
                               zeta = zeta)
        # fit q to map
        para_q = mod_loc_trn.fit_map(variable = variable,
                             model = 'emp_pdf',
                             eps = eps,
                             log = log_q)
        
        
        # para_all_GPD contains (u, xi, sigma)
        # para_all_q contains (q, ln_f_q)
        para = np.hstack((para_q,para_GPD[:,1:2])) # (q, ln_f_q, xi)
        print('Done')
        
# =============================================================================
# Run interpolation        
# =============================================================================

        # interpolate map
        print('Interpolating...', end = '', flush = True)
        try:
            pred_mean, pred_var, hyper = mod_loc_trn.interpolate_map(mod_all_trn.s, para, cov)
        except: 
            print('Error in interpolating - trying again')
            continue
        print('Done')
        
        idx_eval = np.delete(np.arange(mod_all_trn.N_rx),idx, axis = 0)
        d_test = len(idx_eval)

# =============================================================================
# Select rate using multiprocessing
# =============================================================================
        
        R_all = {method : [] for method in methods}
        
        pbar = tqdm(total = d_test, ascii = True, mininterval = 1)
        def update_pbar(*args):
            pbar.update(1)
        
        # Pool with progress bar
        pool = mp.Pool(processes = N_workers)

        # run multiprocessing 
        jobs = []
        for i, idx in enumerate(idx_eval):
            C_all = np.log2(1 + mod_all_trn.get_SNR(idx))
            jobs.append(pool.apply_async(get_R,
                                         args = (C_all, 
                                                 pred_mean[idx], 
                                                 pred_var[idx]),
                                         callback = update_pbar))
        pool.close()
        pool.join()
        pbar.close()
        
        # append results
        for i, job in enumerate(jobs):
            job = job.get()
            for method in methods:
                R_all[method].append(job[method])
                
        R_all = {method : np.vstack(R_all[method]) for method in methods}
            
            
# =============================================================================
# Now compute outage probability, and throughput ratio
# =============================================================================
    
        get_p_out = lambda R, C : np.nanmean(C < R)
        get_throughput = lambda R, R_eps, p_out : (R*(1-p_out))/(R_eps*(1-eps))
        
        
        p_out = {method : np.zeros((d_test, N_n)) for method in methods}
        omega = {method : np.zeros((d_test, N_n)) for method in methods}
        
        mod_all_tst = stat_radio_map(5,path, data_origin = 'APMS',
                              SNR_scale = SNR_scale,
                              fade_subsample= idx_fade_tst)
        
        
        print('Computing performances...', end = '', flush = True)
        for i, idx in enumerate(idx_eval):
            
            C_all = np.log2(1 + mod_all_tst.get_SNR(idx))
            R_eps = np.quantile(C_all, eps)
            
            for method in methods:
                for j in range(N_n):
                    p_out[method][i,j] = get_p_out(R_all[method][i,j], C_all)
                    omega[method][i,j] = get_throughput(R_all[method][i,j],
                                                        R_eps,
                                                        p_out[method][i,j])
            
        print('Done')
    
        for method in methods:
            R_all_L[method].append(R_all[method])
            p_out_L[method].append(p_out[method])
            omega_L[method].append(omega[method])
        idx_eval_L.append(idx_eval)
        l += 1
    
    # stack results
    for method in methods:
        R_all_L[method] = np.vstack(R_all_L[method])
        p_out_L[method] = np.vstack(p_out_L[method])
        omega_L[method] = np.vstack(omega_L[method])
    idx_eval_L = np.hstack(idx_eval_L)
        
    res_all = {'R_all': R_all_L, 'p_out': p_out_L, 'omega': omega_L, 
                'cov': cov, 'log_q': log_q,  'zeta': zeta, 'eps': eps,
                'zeta': zeta,
                'delta': delta, 'SNR_scale': SNR_scale, 
                'd_test': rx_subsample , 'L': L,
                'gibbs_params': gibbs_params,
                'n_rng': n_rng,
                'idx_eval': idx_eval_L}

    save_object(res_all,f'result_files/rate_select_v{version}.pickle')    

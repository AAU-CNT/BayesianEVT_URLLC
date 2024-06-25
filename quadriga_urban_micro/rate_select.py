# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:29:02 2023

@author: Tobias Kallehauge
"""

import sys 
sys.path.insert(0,'../libraries')
data_path = 'generate_quadriga'
from stat_radio_map import stat_radio_map
import numpy as np
import pickle
from rate_select_GPD import rate_select_GPD
from rate_select_non_para import rate_select_non_para
import multiprocessing as mp
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


data_index = 0
version = '' # INSERT VERSION HERE

# parameters
eps = 1e-4
delta = 0.05 # 5%
m = int(1e6) # how many samples per location in previous dataset
d = 500 
d_test = 200
L  = 50
N_ref = int(1e8) 
fade_sim_max_memmory = int(1e6) # maximum samples to simulate in memmory at a time
methods = ['freq_nonpara', 'bay_nonpara', 'freq_GPD', 'bay_GPD']

# model parameters
SNR_scale = 1e12
variable = 'capacity'

# parameters for radio map
cov = ['Matern', 'Matern', 'Matern'] # q, ln_f_q, xi
log_q = True

# parameters for GPD
zeta = 2e-3 # fraction of samples to use to mode tail
N_min = 100 # minium number of samples used in the tail

# gibbs parameters
gibbs_params = {'T': 10_000, 'burn_in': 1000,
                'calibrate_proposal' : True,
                'verbose': False}

# number of samples avaialble for rate sectionon
n_rng = np.hstack([0, 10**(np.linspace(2,6,9))]).astype('int')
N_n = len(n_rng)

# multiprocessing parameters
N_workers = 5 # increase if possible
    
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
        
def get_res(R_idx, mod, idx):
    
    p_out = {method : np.zeros_like(R_idx[method]) for method in methods}
    
    n_chunks = N_ref//fade_sim_max_memmory
    assert n_chunks*fade_sim_max_memmory == N_ref, 'N_ref must be a multible of fade_sim_max_memmory'
    
    for i in range(n_chunks):
        C = np.log2(1 + mod.generate_SNR(idx,int(fade_sim_max_memmory)))
        if i == 0:
            C_eps = np.quantile(C,eps) # can only be based on a single subset but accuracy is not so important here
        for method in methods:
            p_out[method] += np.sum(C[None,None,:] < R_idx[method][:,:,None], axis = 2) 
    
    omega = dict.fromkeys(methods)
    
    for method in methods:
        p_out[method] = p_out[method]/N_ref
        omega[method] =(R_idx[method]*(1-p_out[method]))/(C_eps*(1-eps))
    return(p_out, C_eps, omega)


if __name__ == '__main__':
    mod_all = stat_radio_map(data_index,data_path, SNR_scale = SNR_scale)
    np.random.seed(72)
    R_all_L = {method : [] for method in methods}
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
        idx_fade_all = np.arange(mod_all.N_fade)
        np.random.shuffle(idx_fade_all)
        idx_fade_trn = idx_fade_all[:m]

        mod = stat_radio_map(data_index,data_path, SNR_scale = SNR_scale,
                          rx_subsample_mode = 'random_non_uniform', 
                          rx_subsample= d + d_test)
        s_trn = mod.s[:d]
        s_tst = mod.s[d:]
        
        mod_trn = stat_radio_map(data_index,data_path, SNR_scale = SNR_scale,
                                  rx_subsample_mode = 'list', 
                                  rx_subsample = s_trn,
                                  fade_subsample= idx_fade_trn)
        mod_tst = stat_radio_map(data_index,data_path, SNR_scale = SNR_scale,
                                  rx_subsample_mode = 'list', 
                                  rx_subsample= s_tst)
        
# =============================================================================
#   Fit map
# =============================================================================
        print('Fitting parameters...', end = '', flush = True)
        para_GPD = mod_trn.fit_map(variable = variable,
                                    model = 'GPD',
                                    zeta = zeta,
                                    N_min = N_min)
        # fit q to map
        para_q = mod_trn.fit_map(variable = variable,
                                  model = 'emp_pdf',
                                  eps = eps,
                                  log = log_q)

        para = np.hstack((para_q,para_GPD[:,1:2])) # (q, ln_f_q, xi)
        print('Done')
        
# =============================================================================
# Run interpolation        
# =============================================================================

        # interpolate map
        print('Interpolating...', end = '', flush = True)
        try:
            pred_mean, pred_var, hyper = mod_trn.interpolate_map(mod_tst.s,
                                                                  para, 
                                                                  cov)
        except: 
            print('Error in interpolating - trying again')
            continue
        print('Done')

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
        for i in range(d_test):
            C_all = np.log2(1 + mod_tst.get_SNR(i))
            jobs.append(pool.apply_async(get_R,
                                          args = (C_all, 
                                                  pred_mean[i], 
                                                  pred_var[i]),
                                          callback = update_pbar))
        pool.close()
        pool.join()
        pbar.close()
        
        # append results
        for i, job in enumerate(jobs):
            job = job.get()
            for method in methods:
                R_all[method].append(job[method])
        
        # stack results 
        for method in methods:
            R_all_L[method].append(np.vstack(R_all[method]))
        
            
        idx_eval = []
        for s in mod_tst.s:
            idx_eval.append(np.linalg.norm(s - mod_all.s,axis = 1).argmin())
        idx_eval_L.append(idx_eval)
        
        l += 1
    

    # stack results
    for method in methods:
        R_all_L[method] = np.vstack(R_all_L[method])
    idx_eval_L = np.hstack(idx_eval_L)

    # save results before evaluating outage probabilities
    res_all = {'R_all': R_all_L,
                'cov': cov, 'log_q': log_q,  'zeta': zeta, 
                'eps': eps, 'delta': delta, 'SNR_scale': SNR_scale, 
                'd_test': d_test, 'd': d, 'L': L,
                'data_index': data_index, 'gibbs_params': gibbs_params,
                'idx_eval': idx_eval_L,
                'n_rng': n_rng,
                'N_min': N_min}

# =============================================================================
# Now compute outage probability
# =============================================================================

    idx_test = np.unique(idx_eval_L) # all indicies in the test set
    

    pbar = tqdm(total = len(idx_test), ascii = True, mininterval = 1)
    def update_pbar(*args):
        pbar.update(1)
   
    # Pool with progress bar
    pool = mp.Pool(processes = N_workers)

    # run multiprocessing 
    R_idx_all = {method : [] for method in methods}

    print('Computing results', flush = True)
    time.sleep(0.1)
    jobs = []
    
    for idx in idx_test:
        R_idx = {}
        for method in methods:
           R_idx_method = R_all_L[method][idx_eval_L == idx]
           R_idx[method] = R_idx_method
           R_idx_all[method].append(R_idx_method)
        
        jobs.append(pool.apply_async(get_res,
                                      args = (R_idx,mod_all,idx),
                                      callback = update_pbar))
    pool.close()
    pool.join()
    pbar.close()

    
    # append results
    p_out_all = {method : [] for method in methods}
    omega = {method : [] for method in methods}
    R_eps = np.zeros(len(idx_test))
    for i, job in enumerate(jobs):
        job = job.get()
        R_eps[i] = job[1]
        for method in methods:
            p_out_all[method].append(job[0][method])
            omega[method].append(job[2][method])
    
    # stack results 
    for method in methods:
        p_out_all[method] = np.vstack(p_out_all[method])
        omega[method] = np.vstack(omega[method])
        R_idx_all[method] = np.vstack(R_idx_all[method])
        
    idx_eval_L.sort() # since the order of R was changed based on the order
 
    res_all = {'R_all': R_idx_all, 'p_out': p_out_all, 'omega': omega, 
                'R_eps': R_eps,
                'cov': cov, 'log_q': log_q,  'zeta': zeta, 
                'eps': eps, 'delta': delta, 'SNR_scale': SNR_scale, 
                'd_test': d_test, 'd': d, 'L': L,
                'data_index': data_index, 'gibbs_params': gibbs_params,
                'idx_eval': idx_eval_L,
                'm' : m,
                'n_rng': n_rng}
    
    save_object(res_all, f'result_files/rate_select_{data_index}_v{version}.pickle')    

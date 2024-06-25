# Bayesian_EVT_URLLC

This repository contains the code for the simulations used in the article "Prediction of Rare Channel Conditions using Bayesian Statistics and Extreme Value Theory," which is currently under review for IEEE Transactions on Wireless Communications.

## Content
 - `libraries' : Contains all implemented inference methods used in the paper. See in particular
   - `stat_radio_map.py' : Class for handling data and spatial prediction of channel statistics via CDI maps.
   - `rate_select_non_para.py' : Function for non-parametric rate selection both through a frequentist and Beyesian approach. The scipt also contains some helper functions.
   - `rate_select_GPD' : Functions for rate selection based on the generalized Pareto distribution (GPD) and the theorem from extreme value threory (EVT).
   - `gibbs_GPD.py' : Class for running Markov Chain based on the Metropolis within Gibbs algorithm described in Alg. 1 in the paper.
   - `GP.py' : Class for inference and prediction using Gaussian processes (GP).
 - `quadriga_urban_micro/' : Scripts to generate results from Sec. VII.C based on simulations from QuaDriGa of a 3GPP urban microcell scenario.
   - `generate_quadriga' : Files used to simulate impulse responses. The simulations require that QuaDriGa  is "installed" in the same folder under a folder called "Lisenced". Go to https://quadriga-channel-model.de/ to download QuaDriga for free.
     - `Distribution_Map_generator.m' : Run this script to generate data
   - `result_files/' : Folder for results. Contains results of selected rates from the paper as pickle files. 
   - `find_manual_threshold.py' : Scipt to find the threshold manually via the mean deficit plot, which can then be used for the Heuristic described in App. A.
   - `fit_map.py' : Scipt to spatially interpolate statistics via CDI maps based on GPs and generate Fig. 1 shown in the paper. 
   - `rate_select.py' : Scipt to select rates using the two Bayesian approaches and the baselines. Warning: This script uses multiprocessing and is very computationally heavy, e.g., can take several days depeneding on the avilable number of CPUs.
   - `show_results.py' : Script to generate Figs. 2-4 in the paper based on selected rates.
   - `show_results.py' : Script to generate Fig. 5 in the paper based on selected rates.
 - `experimental_rich_scat' : Scripts to generate results from Sec. VII.D based on experiemental data in a rich scattering environment. 
   - `result_files/' : Folder for results. Contains results of selected rates from the paper as pickle files.
   - `setup_data.py' : Scipt to process raw data into h5 file used for the results. Note that the raw data is not included here.
   - `find_manual_threshold.py' : Scipt to find the threshold manually via the mean deficit plot, which can then be used for the Heuristic described in App. A.
   - `fit_map.py' : Scipt to spatially interpolate statistics via CDI maps based on GPs and generate Fig. 6 shown in the paper. 
   - `rate_select.py' : Scipt to select rates using the two Bayesian approaches and the baselines. Warning: This script uses multiprocessing and is very computationally heavy, e.g., can take several days depeneding on the avilable number of CPUs.
   - `show_results.py' : Script to generate Figs. 7-8 in the paper based on selected rates.
 - `plots/' : Folder for plots in the paper. 
    
## Result files
Some of the data files required for the QuaDriGa are no are not included in the repository directly (due to large file sizes) but can be created by running the appropriate scripts. Note that some of the scripts are computationally heavy and based on multiprocessing. Please contact <tkal@es.aau.dk> to acquire the result files directly. 
The data files required from the expeirmental data are not included due to licencing, but we hope to publish the data soon. 
       
## Dependencies
The simulations are made with Python version `3.11.6`. See requirements.txt for required packages. 

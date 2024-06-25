# BayesianEVT_URLLC

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
    
## Result files
Some of the data files required to produce the results are not included in the repository directly (due to large file sizes) but can be created by running the appropriate scripts. Note that some of the scripts are computationally heavy and based on multiprocessing. Please contact <tkal@es.aau.dk> to acquire the result files directly. 
       
## Dependencies
The simulations are made with Python version `3.10.9`. See requirements.txt for required packages. 

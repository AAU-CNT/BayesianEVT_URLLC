%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate a map of distributions. Given a scenario size and a
% position for the Tx, computes the geometric channel using Quadriga
% simulator in a mesh of points. Then, adds realizations of fast fading on
% top of the geometric channel for each point in the mesh. The script can
% save the channel object in a .mat file (for reproducibility), as well as
% the geometric channel map and the distribution map in h5 dataset format. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------------------------------------------------------------------
% Initialization
%------------------------------------------------------------------------
close all
clc
clear 

%------------------------------------------------------------------------
% Parameters
%------------------------------------------------------------------------

% Coordinates of the transmitter (x, y, z) in meters
xyz_tx = [-50,0,10];

% Coordinates of the receivers (x, y, z) in meters. NOTE: leave empty for
% uniformly sampled map according to sample_distance

% Scenario size array [x_min ; x_max; y_min; y_max] in meters
size_s = [-50; 50; -50; 50];

% Users height in meters (for simplicity, it is assumed the same for all)
% ONLY if xyz_rx is an empty matrix
user_z = 1.5;

sample_mode = "uniform"; 

if sample_mode == "uniform" % select sample distance
    sample_distance = 2; % Sample distance in maps (in meters)
    xyz_rx = [];
elseif sample_mode == "thomas" % select process parameters
    lambdaParent = 0.005; % parent process rate
    lambdaDaughter = 5; % daughter process rate
    sigma = 3.5; % daughter process standard deviation from parent

    xyz_rx = rthomas(size_s(1),size_s(2),size_s(3),size_s(4), user_z, ...
                     lambdaParent,lambdaDaughter,sigma); 
end 

% Frequency of operation      
frequency = 3.6e9;

% Number of samples of independent fading 
Nsamples = 1e6;

% Channel type 
type = '3GPP_3D_UMi_LOS'; % see pg 81 of Quadriga documentation

% Fading type
fade_type = "random"; % either angle or random 

% Save original H matrix?
save_h = false;

% Fast fading correlation distance [m]
SC_lambda = 5; 

draw_power_map = false; 
    
% Number of active clusters
NumClusters = 20; 

% Name of the dataset (it is the same for the object and the datasets)
name_index = 0;  
filename = sprintf('./Stored/Distribution_map_%d',name_index);

% For reproducibility
rng(name_index);


%------------------------------------------------------------------------
% Geometric Channel Generation
%------------------------------------------------------------------------

% Coordinates of the users (in case that xyz_rx is an empty matrix)
if (isempty(xyz_rx))
    [XX,YY] = ndgrid(size_s(1):sample_distance:size_s(2),...
                     size_s(3):sample_distance:size_s(4));
    xyz_rx = [XX(:), YY(:), user_z*ones(numel(XX),1)];
end

% Create channel object
ch = Channel(xyz_tx, xyz_rx, frequency, type, draw_power_map, size_s, ...
               sample_distance, SC_lambda, NumClusters);
           
% Calculate geometric channel + power map if draw_power_map == true. 
% Geometric channel is only computed at positions indicated by xyz_rx.
% Power map is generated in the whole scenario size_s at sample_distance
ch.draw();

%------------------------------------------------------------------------
% Store configuration file in .csv (File 1)
%------------------------------------------------------------------------
labels = {type; 'x_min'; 'x_max'; 'y_min'; 'y_max'; 'sample_distance'; ...
          'SC_lambda'; 'NumClusters'; 'Nx_power_map'; 'Ny_power_map' ;...
          'x_tx';  'y_tx'; 'z_tx'; 'frec'; 'N_points_dist_map';...
          'N_samples_fading'; 'Fading type'};
values = [0; size_s(1); size_s(2); size_s(3); size_s(4);...
          sample_distance; SC_lambda; NumClusters; ...
          length(ch.power_map.x_pos); length(ch.power_map.y_pos); ...
          xyz_tx(1); xyz_tx(2); xyz_tx(3); frequency; size(xyz_rx,1);...
          Nsamples; fade_type];
      
 % Create directory if it does not exist
 if ~exist('Stored', 'dir')
     mkdir('Stored')
 end
 
 % Create and save config file
 writetable(table(labels,values),...
     [filename '_config.csv'],'WriteVariableNames',0); 
 
%------------------------------------------------------------------------
% Store power map in .hdf5 (File 2)
%------------------------------------------------------------------------
% Remove old dataset
if exist([filename '_radio_map.h5'],'file')
    delete([filename '_radio_map.h5']);
end

%--------------------------------------------------------------------------
% Save coordinates in h5 file
%--------------------------------------------------------------------------

h5create([filename '_radio_map.h5'],'/ue_coordinates', size(xyz_rx));
h5write([filename '_radio_map.h5'],'/ue_coordinates', xyz_rx);

%------------------------------------------------------------------------
% Fast fading generation for specified points and storage. 
% Data is generated and storaged by chunks so we reduce the RAM 
% requirements
%------------------------------------------------------------------------
% Store original channel in .mat
if save_h
    H_org = ch.H;
    save([filename '_original_channel.mat'], 'H_org'); 
end

% save coefficients for each path


% Chunk and total size
chunk_size = min(10, size(xyz_rx,1)); 
num_points = size(xyz_rx,1);

% Create h5 dataset for distribution map
h5create([filename '_radio_map.h5'],'/fading_samples', ...
    [Nsamples size(xyz_rx,1)], 'ChunkSize', [Nsamples chunk_size]);

% Temporal matrix for storage
temp = zeros(chunk_size, Nsamples);

% Loop to reduce RAM requirements
f = waitbar(0, 'Generating fading');
for k_chunk = 1:chunk_size:num_points
    waitbar(k_chunk/num_points, f, sprintf('Generating fading: %d %%', floor(k_chunk/num_points*100)));
    range = k_chunk:min(k_chunk + chunk_size-1,num_points);
    for k_in = range
        temp(k_in-k_chunk+1,:) = ...
            abs(sum(squeeze(ch.gen_channel_samples(Nsamples, k_in, fade_type)),1)).^2;
    end
    % Store power channel coefficients in dataset
    h5write([filename '_radio_map.h5'],'/fading_samples', ...
        temp(1:length(range),:).', ...
        [1 range(1)], [Nsamples length(range)]);

end
close(f)

%--------------------------------------------------------------------------
% Save channel coefficients and delays in same h5 file
%--------------------------------------------------------------------------

% put all coefficients and delays in tensors with dimensions N_rx, N_tx,
% N_paths
coeff = zeros(num_points, NumClusters); 
delay = zeros(num_points, NumClusters); 
for j = 1:num_points
    idx = j;
    coeff(j,:) = reshape(ch.H(1,idx).coeff, 1, NumClusters); 
    delay(j,:) = reshape(ch.H(1,idx).delay, 1, NumClusters); 
end 


% save coef (real then imaginary)
h5create([filename '_radio_map.h5'],'/coeff_real', [num_points, NumClusters]);
h5write([filename '_radio_map.h5'],'/coeff_real', real(coeff));
h5create([filename '_radio_map.h5'],'/coeff_imag', [num_points, NumClusters]);
h5write([filename '_radio_map.h5'],'/coeff_imag', imag(coeff));

% save delay
h5create([filename '_radio_map.h5'],'/delay', [num_points, NumClusters]);
h5write([filename '_radio_map.h5'],'/delay', delay);














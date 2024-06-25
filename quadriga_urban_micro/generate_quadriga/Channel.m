classdef Channel < handle
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % PROPERTIES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    properties
        type                % see pg 81 in Quadriga documentation for 
                            % types of scenarios 
        verbose = true;     % Boolean to show status bars 
        draw_power_map      % If true, generates power map
        power_map_param = struct('sample_distance', 0.5, 'size', zeros(4,1), ...
                 'rx_height', 1.5, 'tx_power', 0, 'i_freq', 1); 
        SC_lambda     % [m] Correlation distance for small scale fading
        NumClusters   % [] Number of active clusters
    end
    properties(Constant)
        c = 2.99792458e8     % m/s
        root = '.'      % Root directory 
        quadriga_dir = '/Licensed/quadriga_src/'
        licensed_dir = '/Licensed/'
        aux_dir = '/Auxiliary/' % Auxiliary files
    end
    properties(GetAccess=public,SetAccess=private)
        lambda = -1;
        nRx = 0;
        nTx = 0;
        H = [];
        power_map = struct('RSS', [], 'x_pos', [], 'y_pos', []); % Power map struct
    end
    properties(Dependent)
        xyz_tx        % [x1, y1, z1; x2, y2, z2; ...] for each element
        xyz_rx        % [x1, y1, z1; x2, y2, z2; ...] for each element
        frequency     % [Hz]
    end
    properties(Access=private)
        Pxyz_tx = [];
        Pxyz_rx = [];
        Pfrequency = 0;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % METHODS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    methods
        %----------------------------------------------------------------
        % Constructor 
        %----------------------------------------------------------------
        function obj = Channel(xyz_tx, xyz_rx, frequency, type, ...
                draw_power_map, scenario_size, sample_distance,...
                SC_lambda, NumClusters)
            
            %%%%%%%%%% Checking inputs and assigning variables %%%%%%%%%
            obj.xyz_tx = xyz_tx;
            obj.xyz_rx = xyz_rx;
            obj.frequency = frequency;
            obj.type = type;
            obj.draw_power_map = draw_power_map;
            obj.SC_lambda = SC_lambda;
            obj.NumClusters = NumClusters;
            
            % Scenario size
            if obj.draw_power_map
                obj.power_map_param.size = scenario_size;
                obj.power_map_param.sample_distance = sample_distance;
            end          
            %%%%%%%% End of checking inputs and assigning variables %%%%%%
            
            % Adding path 
            addpath([obj.root, obj.quadriga_dir]);
        end
        
        %----------------------------------------------------------------
        % Draw channels -> Calculate channel realization for each user 
        % according to the geometry of the generated scenario (randomly 
        % placed scatterers and distribution of parameters based on 
        % obj.type)
        %----------------------------------------------------------------
        function [H] = draw(obj)
            
            % If there's no channel
            if isempty(obj.H)
                disp('Generating channel...')
                obj.simulate_channel_quadriga();
            end
            
            % Assign channel to property 
            H = obj.H;
            
        end

        %----------------------------------------------------------------
        % Simulate Quadriga Channel -> Generate channel realization for
        % each link Tx-Rx. Only one sample is generated according to the 
        % specific geometry, i.e., phases of the paths are determined 
        % geometrically. 
        % The ouput is the channel tensor obj.H with the channel
        % coefficient for each path between each Tx-Rx of dimensions:
        % [num_ant_Rx, num_ant_Tx, num_Paths, 1]. 
        % If draw_power_map is enabled, it also calculates the RSS map for
        % the scenario size indicated. 
        %----------------------------------------------------------------
        function simulate_channel_quadriga(obj)

            % Load simulation parameters
            s = qd_simulation_parameters;
            s.sample_density = 2.1;             % Number of samples per half-wavelength
            s.use_absolute_delays = 1;          % Include delay of LOS path
            s.center_frequency = obj.frequency;
            s.show_progress_bars = obj.verbose;

            % Create tracks and antennas for all actors and put them into layout
            tx_track = qd_track('linear'); tx_track(1) = []; % create array to place objects.
            tx_array = qd_arrayant(); tx_array(1) = []; % create array to place objects.
            for n = 1:obj.nTx
                trackLength = 0;
                trackDirection = deg2rad(0);
                tx_track(n) = qd_track('linear', trackLength, trackDirection);
                tx_track(n).initial_position = obj.xyz_tx(n,:).';
                tx_track(n).name = ['tx',num2str(n)];

                tx_array(n) = qd_arrayant('dipole');

                tx_array(n).center_frequency = obj.frequency;
                tx_array(n).element_position = obj.xyz_tx.' - obj.xyz_tx(n,:).';

            end

            rx_track = qd_track('linear'); rx_track(1) = []; % create array to place objects.
            rx_array = qd_arrayant(); rx_array(1) = []; % create array to place objects.
            for n = 1:obj.nRx
                % All actors start facing the x-axis at the randomized intial_position.
                trackLength = 0;
                trackDirection = deg2rad(0);
                rx_track(n) = qd_track('linear',trackLength,trackDirection);
                rx_track(n).initial_position =obj.xyz_rx(n,:).'; 
                rx_track(n).name = ['rx',num2str(n)];

                rx_array(n) = qd_arrayant('dipole');
                rx_array(n).center_frequency = obj.frequency;
                rx_array(n).element_position = zeros(3,1);
            end

            % Assign type of scenario
            scenario = obj.type; 

            l = qd_layout( s );             % create a network layout
            l.rx_track = rx_track;
            l.tx_track = tx_track;
            l.rx_array = rx_array;
            l.tx_array = tx_array;
            l.set_scenario(scenario,[],1);
            
            % Set spatial correlation
            b_spat = l.init_builder;
            %b_spat.scenpar.PerClusterDS = 0;
            %b_spat.scenpar.KF_sigma = 2; 
            b_spat.scenpar.SC_lambda = obj.SC_lambda;
            b_spat.scenpar.NumClusters = obj.NumClusters; 

            % If power map option is activated
            if obj.draw_power_map

                % Generate radio map
                [ map, x_coords, y_coords] = l.power_map(scenario, 'detailed',...
                    obj.power_map_param.sample_distance, ...
                    obj.power_map_param.size(1), ...
                    obj.power_map_param.size(2), ...
                    obj.power_map_param.size(3),...
                    obj.power_map_param.size(4),...
                    obj.power_map_param.rx_height,...
                    obj.power_map_param.tx_power,...
                    obj.power_map_param.i_freq);

                % Save power map
                obj.power_map.RSS = map{1};
                obj.power_map.x_pos = x_coords;
                obj.power_map.y_pos = y_coords;                

            end

            % Simulate channels
            %[obj.H, obj.h_builder] = l.get_channels(); 
            b_spat.gen_ssf_parameters;
            obj.H = get_channels(b_spat);
            
        end     
        
        %----------------------------------------------------------------
        % Generate Nsamples for user k using the amplitudes determined by
        % the geometrical calculation and stored in obj.H, but adding
        % a phase shift according to random movement of the users
        %----------------------------------------------------------------
        function [h_rv] = gen_channel_samples(obj, Nsamples, user, phase)
            
            % Checking valid user number
            if (user <= 0 || user > obj.nRx)
               error('Invalid user index.')
            end
            
            % Checking valid number of samples
            if (Nsamples <= 0 || Nsamples ~= round(Nsamples))
                error('Nsamples must be a positive integer number.')
            end
            
            if phase == "angle"
                % Get the corresponding channel coefficients
                channel = squeeze(obj.H(user).coeff);
                AoA = obj.H(user).par.AoA_cb*pi/180; % Azimuth of arrival [rad]
                PoA = pi/2 - obj.H(user).par.EoA_cb*pi/180; % Polar angle of arrival [rad]
                
                % Generate random movement in the order of the wavelength
                shift = obj.lambda/2*(rand(3,Nsamples) - 0.5); 
                
                % Wavevector
                kvec = 2*pi/obj.lambda*[sin(PoA).*cos(AoA); sin(PoA).*sin(AoA); ...
                    cos(PoA)].';
                
                % Generate channel through steering vectors (plane wave assumption)
                h_rv = abs(channel).*exp(1i*kvec*shift);
            elseif phase == "random"
                % Get the corresponding channel coefficients
                channel = obj.H(user).coeff;
            
                % Generate channel by adding random phases
                h_rv = abs(channel).*exp(1i*2*pi*rand([size(channel) Nsamples]));

                h_rv = squeeze(h_rv); 

            end 
            
        end
        
        %----------------------------------------------------------------
        % Depicts graphically the generated radio map if it exits. 
        %----------------------------------------------------------------
        function plot_power_map(obj)
            
            if (~isempty(obj.power_map.RSS))
            
                % Calculate power in dB
                P_db = 10*log10(sum(obj.power_map.RSS, 4));

                % Represent radio map in color map
                figure()
                plot(obj.xyz_tx(1),obj.xyz_tx(2),'ks','linewidth',2);
                hold on
                imagesc(obj.power_map.x_pos, obj.power_map.y_pos, P_db ); 
                axis(obj.power_map_param.size.');
                plot(obj.xyz_tx(1),obj.xyz_tx(2),'ks','linewidth',2);
                legend('Tx')
                caxis( max(P_db(:)) + [-50 0] );                        % Color range
                colmap = colormap;
                colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
                set(gca,'layer','top')                                  % Show grid on top of the map
                colorbar('south')
                title('Received power [dB]')
                
            else
                error('Power map does not exist. Generate it first.')
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SETS and GETS FUNCTIONS 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function out = get.xyz_tx(obj)
            out = obj.Pxyz_tx;
        end
        function set.xyz_tx(obj,value)
            if any(~( isnumeric(value) && isreal(value) ))
                error('Coordinates must be real numbers')
            elseif ~all( size(value,2) == 3 )
                error('Coordinate must have three columns. Format: [x1, y1, z1; x2, y2, z2]')
            end
            obj.nTx = size(value,1);
            obj.Pxyz_tx = value;
        end
        function out = get.xyz_rx(obj)
            out = obj.Pxyz_rx;
        end
        function set.xyz_rx(obj,value)
            if any(~( isnumeric(value) && isreal(value) ))
                error('Coordinates must be real numbers')
            elseif ~all( size(value,2) == 3 )
                error('Coordinate must have three columns. Format: [x1, y1, z1; x2, y2, z2]')
            end
            obj.nRx = size(value,1);
            obj.Pxyz_rx = value;
        end
        function out = get.frequency(obj)
            out = obj.Pfrequency;
        end
        function set.frequency(obj,value)
            if any(size(value) > 1)
                error('Frequency must be a scalar')
            elseif ~( isnumeric(value) && isreal(value) )
                error('Frequency must be a real number')
            end
            obj.lambda = obj.c/value;
            obj.Pfrequency = value;
        end
    end
end


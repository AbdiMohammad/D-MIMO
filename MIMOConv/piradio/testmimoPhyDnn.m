
% Specify the main folder
mainFolder = '/home/microway/MU-MIMO-PhyDnn/MU-MIMO-Pi-Radio/new_radio';

% Generate the path string for the main folder and all its subfolders
allSubfolders = genpath(mainFolder);

% Add the paths to the MATLAB path
addpath(allSubfolders);

% Save the changes to the MATLAB path (optional)
savepath;

close all;

%% Generate transmit symbols (e.g., 10 sets of random symbols for 8 antennas)
numSymbols = 100; % Number of symbols you want to transmit
transmit_symbols = -10 + (10 + 10) * (rand(numSymbols, 8) + 1i * rand(numSymbols, 8));


% Call the mimoPhyDnn function
[all_received_symbols, all_CFR] = MIMODistNet(transmit_symbols, sdr0, sdr1);

b =3;

    %% Clear all workspace variables except for SDR objects
    varsToKeep = {'sdr0', 'sdr1', 'transmit_symbols', 'n', 'all_CFR', 'all_received_symbols'};
    allVars = whos;
    clearCommand = 'clear ';
    for k = 1:length(allVars)
        if ~ismember(allVars(k).name, varsToKeep)
            clearCommand = [clearCommand allVars(k).name ' '];
        end
    end
    eval(clearCommand);
    clear allVars clearCommand k
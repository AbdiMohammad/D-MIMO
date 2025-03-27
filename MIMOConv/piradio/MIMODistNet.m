function [symbolsRx, CFRs] = MIMODistNet(sdr0, sdr1)

qpskmod = comm.QPSKModulator;
% symbolsTx = reshape(qpskmod(round(3 * rand(40 * 8, 1))), 40, 8);
symbolsTx = (randi([0, 1], 100, 8) * 2 - 1) + 1j * (randi([0, 1], 100, 8) * 2 - 1);
% symbolsTx = repmat(qpskmod(round(3 * rand(40 * 1, 1))), 1, 8);
% Initialize arrays to store all CFR matrices and received symbols
CFRs = []; % This will store all CFR matrices
symbolsRx = []; % This wsill store all received symbols

nStreams = 8;
sdrTx = sdr0;
sdrRx = sdr1;
nSymbols = 1024;
nSkip = nSymbols * 3;
nBatch = 3;

nSync = 32;
nPilot = 64;
nData = nSymbols - nSync - nPilot;

nPackets = ceil(size(symbolsTx, 1) / nData);

% Break the symbolsTx into packets
for packetIdx = 1 : nPackets
    
    %% Generate the SYNC and PILOT    
    sync_real = -10 + (10 + 10) * rand(nSync, nStreams);  % Generate random real parts between -10 and 10
    sync_imag = -10 + (10 + 10) * rand(nSync, nStreams);  % Generate random imaginary parts between -10 and 10
    syncTx = reshape(qpskmod(round(3 * rand(nSync * nStreams, 1))), nSync, nStreams);
    % sync_real + 1i * sync_imag;  % Combine real and imaginary parts
    
    pilot_real = -10 + (10 + 10) * rand(nPilot, nStreams);  % Generate random real parts between -10 and 10
    pilot_imag = -10 + (10 + 10) * rand(nPilot, nStreams);  % Generate random imaginary parts between -10 and 10
    % pilotTx = reshape(qpskmod(round(3 * rand(nPilot * nStreams, 1))), nPilot, nStreams);
    pilotTx = hadamard(nPilot);
    pilotTx = transpose(pilotTx(1 : nStreams, :));
    pilotTx = pilotTx + 1j * pilotTx;
    % pilot_real + 1i*pilot_imag;  % Combine real and imaginary parts
    
    % Prepare txtd with shape (nSymbols, nStreams)
    txtd = zeros(nSymbols, nStreams) + 1i * zeros(nSymbols, nStreams);
    
    txtd(1 : nSync, :) = syncTx;
    txtd(1 + nSync : nSync + nPilot, :) = pilotTx;
    if packetIdx * nData > size(symbolsTx, 1)
        nDataRemaining = size(symbolsTx, 1) - ((packetIdx - 1) * nData);
    else
        nDataRemaining = nData;
    end
    txtd(1 + nSync + nPilot : nSync + nPilot + nDataRemaining, :) = symbolsTx((packetIdx -1) * nData + 1 : (packetIdx -1) * nData + nDataRemaining, :);
    
    % aod = 30;
    % aod = deg2rad(aod);
    % piTx = txtd .* exp(1j * [1 : nStreams] * pi * sin(aod)); % Apply BF

    piTx = sdrTx.applyCalTxArray(txtd);
    % Normalize the energy of the tx array and scale with txPower
    txPower = 8000;
    piTx = txPower * piTx ./ max(abs(piTx(:)));
    
    sdrTx.send(piTx);
    
    for i = 1 : 3
        rxtd = sdrRx.recv(nSymbols, nSkip, nBatch);
    end
    rxtd = sdrRx.applyCalRxArray(rxtd);
    rxtd = squeeze(rxtd(:, 3, :));

    % Get the received version of the sync and pilot symbols
    % syncRx = rxtd(1 : nSync, :);
    
    %% Perform the syncronization
    % Find the start of the packet
    % it should be the same across antennas
    % correlate each rx symbol (sum of all tx symbols) with each tx symbol
    locs = reshape(finddelay(repmat(txtd(1 : nSync, :), 1, nStreams), repelem(rxtd, 1, nStreams)), nStreams, nStreams);

    % Remove outliers
    median_loc = round(median(locs, 'all'));
    locs(abs(locs - median_loc) > 5) = median_loc;
    loc_selected = round(mean(locs, 'all'));

    rxtd = [rxtd(loc_selected + 1 : nSymbols, :); ...
    rxtd(1 : loc_selected, :)];

    %% Estimate the CFR using the pilot symbol
    % Initialize CFR matrix for each symbol
    CFR = zeros(nStreams, nStreams);

    pilotRx = rxtd(1 + nSync : nSync + nPilot, :);  % The pilot symbol for CFR estimation

    %% Equalize the Data symbols using calculated CFR
    dataRx = rxtd(1 + nSync + nPilot : nSync + nPilot + nDataRemaining, :);
    
    CFR = transpose(pilotTx \ pilotRx);
    % CFR_compare = reshape(repmat(pilotRx(1, :), 1, nStreams) ./ repelem(pilotTx(1, :), 1, nStreams), nStreams, nStreams);
    % dataRx_eq = dataRx * (ones(nStreams, nStreams) ./ CFR) / nStreams;
    % dataRx_eq = transpose(CFR \ transpose(dataRx));
    % dataRx_eq = transpose(dataRx) \ CFR;

    % CFR = reshape(kron(pilotTx, eye(nStreams)) \ reshape(transpose(pilotRx), [], 1), nStreams, nStreams);
    % dataRx_eq = transpose(CFR \ transpose(dataRx));
    % figure();
    % hold on;
    % grid on;
    % plot(dataRx_eq, 'b.');
    % plot(symbolsTx, 'r*');

    % Equalize each symbol using its own CFR
    % CFR = reshape(repmat(rxtd, 1, nStreams) ./ repelem(txtd, 1, nStreams), nSymbols, nStreams, nStreams);
    % rxtd_eq = repmat(rxtd, 1, 1, nStreams) ./ CFR;
    % rxtd_eq = squeeze(mean(rxtd_eq, 2));
    % dataRx_eq = rxtd_eq(1 + nSync + nPilot : nSync + nPilot + nDataRemaining, :);
    % figure();
    % plot(dataRx_eq, '.');

    % Equalize all the data symbols using one CFR
    % CFR = reshape(repmat(pilotRx, 1, nStreams) ./ repelem(pilotTx, 1, nStreams), nStreams, nStreams);
    % dataRx_eq = repmat(dataRx, 1, 1, nStreams) ./ repmat(reshape(CFR, 1, nStreams, nStreams), nDataRemaining, 1, 1);
    % dataRx_eq = squeeze(mean(dataRx_eq, 2));
    % figure();
    % hold on;
    % grid on;
    % subplot(1, 2, 1);
    % plot(dataRx_eq, '.');
    % dataRx_eq = dataRx * (ones(nStreams, nStreams) ./ CFR) / nStreams;
    % subplot(1, 2, 2);
    % plot(dataRx_eq, 'b.');
    % plot(symbolsTx, 'r*');

    % dataRx = repmat(dataRx, 1, nStreams) * (ones(nSymbols, nStreams, nStreams) ./ transpose(CFR));
    % for symIdx = 1 : nSymbols
    %     rxtd_eq(symIdx, :) = rxtd(symIdx, :) * (ones(nStreams, nStreams) ./ squeeze(CFR(symIdx, :, :))) / nStreams;
    % end


    % Store CFR and received symbol in their respective arrays
    CFRs = cat(3, CFRs, CFR); % Stack CFR matrices along the 3rd dimension
    symbolsRx = [symbolsRx; dataRx]; % Append received symbol as a new row
end

%% Clear all workspace variables except for SDR objects
varsToKeep = {'sdr0', 'sdr1', 'symbolsTx', 'CFRs', 'symbolsRx'};
allVars = whos;
clearCommand = 'clear ';
for k = 1:length(allVars)
    if ~ismember(allVars(k).name, varsToKeep)
        clearCommand = [clearCommand allVars(k).name ' '];
    end
end
eval(clearCommand);
clear allVars clearCommand k
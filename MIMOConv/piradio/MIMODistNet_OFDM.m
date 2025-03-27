function [sampleCFR] = MIMODistNet_OFDM(sdr0, sdr1)

% qpskmod = comm.QPSKModulator;
% symbolsTx = reshape(qpskmod(round(3 * rand(40 * 8, 1))), 40, 8);
dataTx = (randi([0, 1], 9 * 800, 8) * 2 - 1) + 1j * (randi([0, 1], 9 * 800, 8) * 2 - 1);
% symbolsTx = repmat(qpskmod(round(3 * rand(40 * 1, 1))), 1, 8);
% Initialize arrays to store all CFR matrices and received data symbols
CFRs = []; % This will store all CFR matrices
dataRx = []; % This will store all received data symbols

nStreams = 8;
sdrTx = sdr0;
sdrRx = sdr1;
txPower = 7000;

bufferSize = 1024 * 8;
nCarriers = 1024;
nUsableCarriers = 800;
packet2PilotRatio = 10;

nSkip = bufferSize * 3;
nBatch = 1;
nRetriesRx = 1;

nData = size(dataTx, 1);
nDataBatches = ceil(nData / nUsableCarriers);
nBatches = nDataBatches + ceil(nDataBatches / (packet2PilotRatio - 1));
nBuffers = ceil((nBatches * nCarriers) / bufferSize);
nBatchesInBuffer = floor(bufferSize / nCarriers);

% Break the data into buffers which are copied to Pi-Radio memory in one shot
for bufferIdx = 1 : nBuffers
    fdTx = zeros(nCarriers, nBatchesInBuffer, nStreams);
    % Break the data into batches where each batch consists of nCarriers symbols
    for batchIdxInBuffer = 1 : nBatchesInBuffer
        batchIdx = (bufferIdx - 1) * nBatchesInBuffer + batchIdxInBuffer;
        dataBatchIdx = batchIdx - ceil(batchIdx / packet2PilotRatio);
        if rem(batchIdx, packet2PilotRatio) == 1
            % This is a PILOT batch
            % Generate the PILOT
            % pilot_real = -10 + (10 + 10) * rand(nPilot, nStreams);  % Generate random real parts between -10 and 10
            % pilot_imag = -10 + (10 + 10) * rand(nPilot, nStreams);  % Generate random imaginary parts between -10 and 10
            % pilotTx = reshape(qpskmod(round(3 * rand(nPilot * nStreams, 1))), nPilot, nStreams);
            % pilot_real + 1i*pilot_imag;  % Combine real and imaginary parts
            % symbolsTx = hadamard(nCarriers);
            % symbolsTx = transpose(symbolsTx(1 : nStreams, :));
            % symbolsTx = symbolsTx + 1j * symbolsTx;
            symbolsTx = (randi([0, 1], nUsableCarriers, nStreams) * 2 - 1) + 1j * (randi([0, 1], nUsableCarriers, nStreams) * 2 - 1);
            nSymbolsBatch = nUsableCarriers;
        else
            if dataBatchIdx * nUsableCarriers > nData
                nSymbolsBatch = nData - ((dataBatchIdx - 1) * nUsableCarriers);
            else
                nSymbolsBatch = nUsableCarriers;
            end

            symbolsTx = dataTx((dataBatchIdx - 1) * nUsableCarriers + 1 : (dataBatchIdx - 1) * nUsableCarriers + nSymbolsBatch, :);
        end

        fdTx((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) - (nUsableCarriers / 2) + nSymbolsBatch, batchIdxInBuffer, :) = ...
            symbolsTx(1 : nSymbolsBatch, :);
    end

    fdTx = fftshift(fdTx, 1);
    piTx = ifft(fdTx, [], 1);

    piTx = reshape(piTx, bufferSize, nStreams);
    piTx = sdrTx.applyCalTxArray(piTx);
    piTx = txPower * piTx ./ max(abs(piTx(:)));
    sdrTx.send(piTx);
    
    for i = 1 : nRetriesRx
        tdRx = sdrRx.recv(bufferSize, nSkip, nBatch);
    end
    tdRx = sdrRx.applyCalRxArray(tdRx);
    tdRx = squeeze(tdRx(:, nBatch, :));
    tdRx = reshape(tdRx, nCarriers, nBatchesInBuffer, nStreams);

    % Process each batch of symbols separately consisting of nCarriers symbols
    for batchIdxInBuffer = 1 : nBatchesInBuffer
        batchIdx = (bufferIdx - 1) * nBatchesInBuffer + batchIdxInBuffer;
        dataBatchIdx = batchIdx - ceil(batchIdx / packet2PilotRatio);

        if rem(batchIdx, packet2PilotRatio) == 1
            % This is a PILOT batch
            % Process the received PILOT for both synchronization and channel
            % estimation
            tdPilotRx = squeeze(tdRx(:, batchIdxInBuffer, :));
            fdPilotRx = fftshift(fft(tdPilotRx, [], 1));

            fdPilotTx = fftshift(squeeze(fdTx(:, batchIdxInBuffer, :)), 1);

            % Perform the synchronization
            % Find the start of the packet
            % it should be the same across antennas
            % correlate each rx symbol (sum of all tx symbols) with each tx symbol
            [~, locs] = max(abs(ifft(repmat(fdPilotTx, 1, nStreams) .* conj(repelem(fdPilotRx, 1, nStreams)), [], 1)), [], 1);
            locs = reshape(locs, nStreams, nStreams);
    
            % Remove outliers
            median_loc = round(median(locs, 'all'));
            locs(abs(locs - median_loc) > 5) = median_loc;
            loc_selected = round(mean(locs, 'all'));

            tdPilotRx = [tdPilotRx(nCarriers - loc_selected + 2 : nCarriers, :); ...
                tdPilotRx(1:nCarriers - loc_selected + 1, :)];
            fdPilotRx = fftshift(fft(tdPilotRx, [], 1));

            % Perform the channel estimation
            CFR = zeros(nCarriers, nStreams, nStreams);
            CFR((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) + (nUsableCarriers / 2), :, :) = reshape(repmat(fdPilotRx((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) + (nUsableCarriers / 2), :), 1, nStreams) ./ repelem(fdPilotTx((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) + (nUsableCarriers / 2), :), 1, nStreams), nUsableCarriers, nStreams, nStreams);
            % Store CFRs
            CFRs = cat(4, CFRs, CFR); % Stack CFR matrices along the last (4th) dimension
        else
            tdDataRx = squeeze(tdRx(:, batchIdxInBuffer, :));
            tdDataRx = [tdDataRx(nCarriers - loc_selected + 2 : nCarriers, :); ...
                tdDataRx(1:nCarriers - loc_selected + 1, :)];

            fdDataRx = fftshift(fft(tdDataRx, [], 1));

            if dataBatchIdx * nUsableCarriers > nData
                nSymbolsBatch = nData - ((dataBatchIdx - 1) * nUsableCarriers);
            else
                nSymbolsBatch = nUsableCarriers;
            end
            % symbolsRx = tensorprod(fdDataRx((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) - (nUsableCarriers / 2) + nSymbolsBatch, :), ...
            %     ones(nSymbolsBatch, nStreams, nStreams) ./ CFR((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) - (nUsableCarriers / 2) + nSymbolsBatch, :, :), 2) / nStreams;
            symbolsRx = repmat(fdDataRx((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) - (nUsableCarriers / 2) + nSymbolsBatch, :), 1, 1, nStreams) ./ ...
                CFR((nCarriers / 2) - (nUsableCarriers / 2) + 1 : (nCarriers / 2) - (nUsableCarriers / 2) + nSymbolsBatch, :, :);
            symbolsRx = squeeze(mean(symbolsRx, 2));
            dataRx = [dataRx; symbolsRx]; % Append received symbols as a new row
        end
    end
end

figure();
hold on;
subplot(1, 2, 1);
plot(dataTx, "b*");
subplot(1, 2, 2);
plot(dataRx, "r.");
% sampleCFR = squeeze(CFRs((nCarriers / 2), :, :));

end
        % %% Estimate the CFR using the pilot symbol
        % % Initialize CFR matrix for each symbol
        % CFR = zeros(nStreams, nStreams);
        % 
        % pilotRx = tdRx(1 + nSync : nSync + nPilot, :);  % The pilot symbol for CFR estimation
        % 
        % %% Equalize the Data symbols using calculated CFR
        % dataRx = tdRx(1 + nSync + nPilot : nSync + nPilot + nDataRemaining, :);
        % 
        % CFR = transpose(symbolsTx \ pilotRx);
        % % CFR_compare = reshape(repmat(pilotRx(1, :), 1, nStreams) ./ repelem(pilotTx(1, :), 1, nStreams), nStreams, nStreams);
        % % dataRx_eq = dataRx * (ones(nStreams, nStreams) ./ CFR) / nStreams;
        % % dataRx_eq = transpose(CFR \ transpose(dataRx));
        % % dataRx_eq = transpose(dataRx) \ CFR;
        % 
        % % CFR = reshape(kron(pilotTx, eye(nStreams)) \ reshape(transpose(pilotRx), [], 1), nStreams, nStreams);
        % % dataRx_eq = transpose(CFR \ transpose(dataRx));
        % % figure();
        % % hold on;
        % % grid on;
        % % plot(dataRx_eq, 'b.');
        % % plot(symbolsTx, 'r*');
        % 
        % % Equalize each symbol using its own CFR
        % % CFR = reshape(repmat(rxtd, 1, nStreams) ./ repelem(txtd, 1, nStreams), nSymbols, nStreams, nStreams);
        % % rxtd_eq = repmat(rxtd, 1, 1, nStreams) ./ CFR;
        % % rxtd_eq = squeeze(mean(rxtd_eq, 2));
        % % dataRx_eq = rxtd_eq(1 + nSync + nPilot : nSync + nPilot + nDataRemaining, :);
        % % figure();
        % % plot(dataRx_eq, '.');
        % 
        % % Equalize all the data symbols using one CFR
        % % CFR = reshape(repmat(pilotRx, 1, nStreams) ./ repelem(pilotTx, 1, nStreams), nStreams, nStreams);
        % % dataRx_eq = repmat(dataRx, 1, 1, nStreams) ./ repmat(reshape(CFR, 1, nStreams, nStreams), nDataRemaining, 1, 1);
        % % dataRx_eq = squeeze(mean(dataRx_eq, 2));
        % % figure();
        % % hold on;
        % % grid on;
        % % subplot(1, 2, 1);
        % % plot(dataRx_eq, '.');
        % % dataRx_eq = dataRx * (ones(nStreams, nStreams) ./ CFR) / nStreams;
        % % subplot(1, 2, 2);
        % % plot(dataRx_eq, 'b.');
        % % plot(symbolsTx, 'r*');
        % 
        % % dataRx = repmat(dataRx, 1, nStreams) * (ones(nSymbols, nStreams, nStreams) ./ transpose(CFR));
        % % for symIdx = 1 : nSymbols
        % %     rxtd_eq(symIdx, :) = rxtd(symIdx, :) * (ones(nStreams, nStreams) ./ squeeze(CFR(symIdx, :, :))) / nStreams;
        % % end
%% Clear all workspace variables except for SDR objects
% varsToKeep = {'sdr0', 'sdr1', 'dataTx', 'CFRs', 'dataRx'};
% allVars = whos;
% clearCommand = 'clear ';
% for k = 1:length(allVars)
%     if ~ismember(allVars(k).name, varsToKeep)
%         clearCommand = [clearCommand allVars(k).name ' '];
%     end
% end
% eval(clearCommand);
% clear allVars clearCommand k
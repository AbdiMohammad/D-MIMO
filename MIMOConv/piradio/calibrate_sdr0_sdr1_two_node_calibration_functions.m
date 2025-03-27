
% clc;
% close all;
% clear all;
% pause(2);

%% TWO-NODE CALIBRATIONS

% In this example, sdr0 is the node under calibration.
% The "helper" is sdr1.

% To perform calibration, first configure the hardware to operate
% at 58 GHz. The self-cal requires this frequency.

%addpath('../');
addpath('helper/');
isDebug = false;		% print debug messages

ip = "10.1.1.43";	% IP Address
sdr0 = piradio.sdr.FullyDigital('ip', ip, 'isDebug', isDebug, ...
    'figNum', 100, 'name', 'lamarr-rev3.1-0001');

ip = "10.1.1.44";	% IP Address
sdr1 = piradio.sdr.FullyDigital('ip', ip, 'isDebug', isDebug, ...
    'figNum', 101, 'name', 'lamarr-rev3.1-0002');

sdr0.fpga.configure('config/rfsoc.cfg');
sdr1.fpga.configure('config/rfsoc.cfg');

sdr0.lo.configure('config/lmx_pdn.txt');
sdr1.lo.configure('config/lmx_pdn.txt');
sdr0.lo.configure('config/lmx_hedy_lamarr_58ghz.txt');
sdr1.lo.configure('config/lmx_hedy_lamarr_58ghz.txt');

% Power down all LTC5594 and HMC630x chips
sdr0.ltc.configure(10, 'config/ltc5594_pdn.txt');
sdr1.ltc.configure(10, 'config/ltc5594_pdn.txt');
sdr0.rffeTx.configure(10, 'config/hmc6300_pdn.txt');
sdr1.rffeTx.configure(10, 'config/hmc6300_pdn.txt');
sdr0.rffeRx.configure(10, 'config/hmc6301_pdn.txt');
sdr1.rffeRx.configure(10, 'config/hmc6301_pdn.txt');

sdr0.set_switches('off');
sdr1.set_switches('off');

% Bring up all TX and RX channels on sdr0
sdr0.ltc.configure(10, 'config/ltc5594_pup.txt');
sdr0.rffeTx.configure(10, 'config/hmc6300_registers.txt');
sdr0.rffeRx.configure(10, 'config/hmc6301_registers.txt');

% Bring up all TX and RX channels on sdr1
sdr1.ltc.configure(10, 'config/ltc5594_pup.txt');
sdr1.rffeTx.configure(10, 'config/hmc6300_registers.txt');
sdr1.rffeRx.configure(10, 'config/hmc6301_registers.txt');

clear ip isDebug;

%%%%%%%%%%%%%%%%%%%%%%
%% CALIBRATION sdr0 %%
%%%%%%%%%%%%%%%%%%%%%%

sdr0 = helper_two_node_calibration(sdr0, sdr1);
%%%%%%%%%%%%%%%%%%%%%%
%% CALIBRATION sdr1 %%
%%%%%%%%%%%%%%%%%%%%%%

sdr1 = helper_two_node_calibration(sdr1, sdr0);

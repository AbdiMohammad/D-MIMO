%
% Company:	New York University
%           Pi-Radio
%
% Engineer: Panagiotis Skrimponis
%           Aditya Dhananjay
%
% Description: This class establishes a communication link over TCP between
% a host and a Xilinx ZynqMP Ultrascale+ RFSoC FPGA board. The host
% configures the RF data converter given a parameter file, receives
% samples from the ADCs, and sends samples to the DACs in non-realtime
% mode. The application running in the PS of the FPGA is based on a
% modified version of Xilinx `rftool`.
%
% Last update on Mar. 5, 2021
%
% Copyright @ 2021
%
classdef RFSoC < matlab.System
    properties
        % Tunable parameters of the RFSoC object
        ip;                 % IP address
        sockData;           % data TCP socket
        sockCtrl;           % ctrl TCP socket
        isDebug = false;	% if 'true' print debug messages
        isComplex = true;   % if 'true' A/D and D/A have
    end
    
    properties (Nontunable)
        % General information for the FPGA design
        nco = 1e9;              % frequency of the NCO in Hz
        nch = 8;                % num of channels
        ndac = 8;               % num of D/A converters
        nadc = 8;               % num of A/D converters
        npar = 2;               % num of parallel sampler per cc
        pll = 3932.16e6;        % base sample rate of the converters in Hz
        drate = 4;              % decimation rate
        irate = 4;              % interpolation rate
        maxTxSamp = int32(32768*8); % store up to 16KB pf tx data
        maxRxSamp = int32(1024^3); % store up to 1GB of rx data
    end
    
    properties (Dependent)
        fs;     % post-decimation/pre-interpolation sample frequency
    end
    
    methods
        function obj = RFSoC(varargin)
            % Constructor
            
            % Set parameters from constructor arguments.
            if nargin >= 1
                obj.set(varargin{:});
            end
            
            % Establish TCP connections.
            obj.connect();
            
            obj.sendCmd("TermMode 1");
        end
        
        function delete(obj)
            % Destructor.
            
            % Close TCP connections.
            obj.disconnect();
        end
        
        function rxtd = recv(obj, nsamp)
            % Initialize the output buffer. The output buffer should have
            % the following dimensions: (nsamp/nch/2) x nch
            rxtd = zeros(nsamp/obj.nadc/2, obj.nadc);
            
            if int32(nsamp*2)> obj.maxRxSamp
                fprintf(2,'Error: max number of rx bytes is %dGB.\r\n', ...
                    obj.maxRxsamp*(1024)^(-3));
                return;
            end
            
            % Read 'nsamp' data from all ADC
            write(obj.sockData, sprintf("ReadDataFromMemory 0 0 %d 0\r\n", 2*nsamp));
            data = reshape(read(obj.sockData, nsamp, 'int16'), [], 1);
            pause(0.1);
            
            % Read the response from the Data TCP Socket
            rsp = read(obj.sockData);
            if (obj.isDebug)
                fprintf(1, "%s", rsp);
            end
            
            % Convert data from 'int16' to 'double'. We reshape the data
            % since the baseband runs at a lower frequency and we process 
            % multiple samples per clock cycle
            data = double(reshape(data, obj.npar, []));
            
            % Create the complex samples for each ADC. The data are being
            % stored sequentially for all ADCs for each I/Q
            for iadc = 1:obj.nadc
                rxtd(:,iadc) = reshape(data(:,(2*iadc-1):2*obj.nadc:end) - ...
                    1j*data(:,2*iadc:2*obj.nadc:end), [], 1);
            end
        end
        
        function send(obj, txtd)
            % First, we need to process the data from the DACs. The
            % expected input to this function is a matrix with dimension
            % (nsamp x nch)
            
            % Convert the complex input data to a 3D matrix with 'int16' values
            tmp = zeros(2, size(txtd,1), size(txtd,2));
            tmp(1,:,:) = int16(imag(txtd));
            tmp(2,:,:) = int16(real(txtd));
            
            % Since the FPGA needs 2 samples of I/Q for every DAC at each
            % clock cyle we need to reshape the tensor
            tmp = reshape(tmp, 2*2, [], obj.ndac);
            
            % We interleave the data for every DAC
            txtd = zeros(2*2, size(txtd,1)*size(txtd,2)/2);
            for idac = 1:obj.ndac
                txtd(:,idac:obj.ndac:end) = reshape(tmp(:,:,idac),2*2,[]);
            end
            
            % Finally, we flatten the tx vector;
            txtd = reshape(txtd,[],1);
            
            nsamp = length(txtd);	% num of samples
            
            if int32(nsamp*2)> obj.maxTxSamp
                fprintf(2,'Error: max number of tx bytes is %dKB.\r\n', ...
                    obj.maxTxSamp/1024);
                return;
            end
            
            % Send the data over TCP with the necessary commands in the
            % control and data channel
            obj.sendCmd(sprintf("LocalMemTrigger 1 0 0 0x0000"));
            write(obj.sockData, sprintf("WriteDataToMemory 0 0 %d 0\r\n", 2*nsamp));
            write(obj.sockData, txtd, 'int16');
            pause(0.1);
            % Read response from the Data TCP Socket
            rsp = read(obj.sockData);
            if (obj.isDebug)
                fprintf(1, "%s", rsp);
            end
            obj.sendCmd("LocalMemTrigger 1 2 0 0x0001");
        end
        
        function configure(obj, file)
            % Parse the output file from the RFDC.
            fid = fopen(file,'r');
            while ~feof(fid)
                tline = fgetl(fid);
                % The following lines parse a file generated from the
                % Xilinx RFDC Windows application:
                %
                % tmp = regexp(tline, '\t', 'split');
                % fprintf(1, '%s\n',tmp{4})
                % obj.sendCmd(tmp{4});
                %
                % However, we are going to parse a simplified version of
                % the file with only the necessary commands.
                if (tline(1) ~= '%')
                    fprintf(1, '%s\n', tline);
                    obj.sendCmd(tline)
                else
                    % If there is a comment, then pause. This will be
                    % helpful to let PLLs stabilize, etc.
                    pause(0.2);
                end
            end
            fclose(fid);
        end
        
        % Create some helper functions
        function fs = get.fs(obj)
            % Return the post-decimation/pre-interpolation sample rate.
            fs = obj.pll/obj.irate;
        end
    end
    
    methods (Access = 'protected')
        function connect(obj)
            % This function establishes communication between a host and
            % an RFSoC device.
            if (isempty(obj.sockData))
                obj.sockData = tcpclient(obj.ip, 8082, "Timeout", 5);
            end
            
            if (isempty(obj.sockCtrl))
                obj.sockCtrl = tcpclient(obj.ip, 8081, "Timeout", 5);
            end
        end
        
        function disconnect(obj)
            % This function disbands the communication sockets between the
            % host and an RFSoC device.
            
            if (~isempty(obj.sockData))
                flush(obj.sockData);
                clear obj.sockData;
            end
            
            if (~isempty(obj.sockCtrl))
                flush(obj.sockCtrl);
                clear obj.sockCtrl;
            end
        end
        
        function sendCmd(obj, cmd)
            % This fu
            % Flush the input/output buffer
            flush(obj.sockCtrl);
            
            % Send a command to the FPGA
            write(obj.sockCtrl, sprintf("%s\r\n",cmd));
            
            % Wait for the FPGA to process the command
            pause(0.1);
            
            % Read response and print in debug mode
            rsp = read(obj.sockCtrl);
            if (obj.isDebug)
                fprintf(1, "%s", rsp);
            end
        end
    end
end

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matlab.engine

class SendOverChannel(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tx_symbols, rx_symbols, channel_matrix):
            ctx.save_for_backward(channel_matrix)
            return torch.cat((rx_symbols.real, rx_symbols.imag), dim=2)
        
        @staticmethod
        def backward(ctx, grad_output):
            channel_matrix = ctx.saved_tensors
            original_shape = grad_output.shape
            grad_output = grad_output.reshape(channel_matrix.shape[0], channel_matrix.shape[1], -1)
            (grad_output_real, grad_output_imag) = grad_output.chunk(2, dim=2)
            grad_input_real = torch.matmul(channel_matrix.real, grad_output_real) - torch.matmul(channel_matrix.imag, grad_output_imag)
            grad_input_imag = torch.matmul(channel_matrix.imag, grad_output_real) + torch.matmul(channel_matrix.real, grad_output_imag)
            grad_input = torch.cat((grad_input_real, grad_input_imag), dim=2)
            grad_input = grad_input.reshape(original_shape)
            return grad_input, None, None

class MIMOChannel(nn.Module):
    def __init__(self, batch_size, n_streams, model="rayleigh", PSNR=20.0) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.N_S = n_streams
        self.PSNR = PSNR
        self.p = 1
        self.register_buffer('channel_matrix', torch.randn(self.batch_size, self.N_S, self.N_S, dtype=torch.complex64))

        self.noise_scale = math.sqrt(0.5 * self.p / math.pow(10, self.PSNR / 10))
        self.model = model

        self.eng = None
        if self.model == "pi-radio":
            assert len(matlab.engine.find_matlab()) > 0, "No running instance of MATLAB is found"
            self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
    
    def set_model(self, model):
        assert model in ["awgn", "rayleigh", "pi-radio", "noiseless-rayleigh"], "MIMO channel model is not supported"
        self.model = model

        if self.model == "pi-radio":
            if self.eng is not None:
                self.eng.quit()
            assert len(matlab.engine.find_matlab()) > 0, "No running instance of MATLAB is found"
            self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
    
    def set_channel_matrix(self, channel_matrix):
        self.channel_matrix = channel_matrix
    
    def update_channel_matrix(self):
        if self.model in ["rayleigh", "noiseless-rayleigh"]:
            self.channel_matrix = torch.randn(self.batch_size, self.N_S, self.N_S, dtype=torch.complex64)
        elif self.model == "pi-radio":     
            if self.eng is None:
                assert len(matlab.engine.find_matlab()) > 0, "No running instance of MATLAB is found"
                self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
            # Send NDP packet
            # rx_samples_matlab, CFRs_matlab = self.eng.mimoPhyDnn(np.random.randn(1, self.N_S), self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=2)
            CFRs_matlab = self.eng.MIMODistNet_OFDM(self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=1)
            self.channel_matrix = torch.tensor(CFRs_matlab)

    def forward(self, x):
        if self.model == "awgn":
            return x + torch.randn_like(x) * self.noise_scale
        elif self.model in ["rayleigh", "noiseless-rayleigh"]:
            assert x.numel() % self.N_S == 0, f"Tensor of length {x.numel} cannot be divided into {self.N_S} parts for different data streams"
            original_shape = x.shape
            x = x.reshape(x.shape[0], self.N_S, -1)
            tx_symbols = torch.complex(*x.chunk(2, dim=2))
            rx_symbols = torch.matmul(self.channel_matrix, tx_symbols)
            x = torch.cat((rx_symbols.real, rx_symbols.imag), dim=2)
            x = x.reshape(original_shape)
            if self.model == "rayleigh":
                return x + torch.randn_like(x) * self.noise_scale
            elif self.model == "noiseless-rayleigh":
                return x
        elif self.model == "pi-radio":
            assert x.numel() % self.N_S == 0, f"Tensor of length {x.numel} cannot be divided into {self.N_S} parts for different data streams"
            original_shape = x.shape
            x = x.reshape(x.shape[0], self.N_S, -1)
            tx_symbols = torch.complex(*x.chunk(2, dim=2))
            rx_symbols_matlab, CFRs_matlab = self.eng.MIMODistNet_OFDM(tx_symbols.detach().cpu().numpy(), self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=2)
            self.channel_matrix = torch.tensor(CFRs_matlab)
            rx_symbols = torch.tensor(rx_symbols_matlab)
            x = SendOverChannel.apply(tx_symbols, rx_symbols, self.channel_matrix)
            x = x.reshape(original_shape)
            return x

class MIMOPrecoder(nn.Module):
    def __init__(self, n_streams, type="task-oriented") -> None:
        super().__init__()
        self.N_S = n_streams
        self.register_buffer('channel_matrix', None)
        self.type = type
        assert self.type in ["task-oriented", "SVD", "ZF", "LMMSE", "MCR2"]
        if self.type == "task-oriented":
            self.linear_weight1 = nn.Parameter(torch.eye(self.N_S * self.N_S, self.N_S * self.N_S, dtype=torch.complex64))
            # self.relu = nn.ReLU()
        
    def set_channel_matrix(self, channel_matrix):
        self.channel_matrix = channel_matrix
    
    def forward(self, x):
        # self.precoding_tensor = torch.view_as_complex(self.relu(torch.view_as_real(torch.matmul(torch.linalg.pinv(self.channel_matrix).flatten(1), self.linear_weight1.T))))
        # self.precoding_tensor = torch.view_as_complex(self.relu(torch.view_as_real(torch.matmul(self.precoding_tensor, self.linear_weight2.T)))).reshape(self.channel_matrix.shape)
        # Only linear layer without any activation function
        if self.type == "task-oriented":
            self.precoding_tensor = torch.matmul(torch.linalg.svd(self.channel_matrix)[2].conj().mT.flatten(1), self.linear_weight1.T).reshape(self.channel_matrix.shape)
            # self.precoding_tensor = torch.matmul(self.precoding_tensor, self.linear_weight2.T)
            # self.precoding_tensor = self.precoding_tensor.unsqueeze(-1).unsqueeze(-1)
        elif self.type == "SVD":
            self.precoding_tensor = torch.linalg.svd(self.channel_matrix)[2].conj().mT
        elif self.type == "ZF":
            self.precoding_tensor = torch.linalg.pinv(self.channel_matrix)
        
        original_shape = x.shape
        x = x.reshape(x.shape[0], self.N_S, -1)
        x = torch.complex(*x.chunk(2, dim=2))
        # x = x.unsqueeze(0).unsqueeze(-1)
        
        x = torch.matmul(self.precoding_tensor, x)
        # x = F.conv2d(x, self.precoding_tensor)
        # x = x.squeeze(0).squeeze(-1)
        
        # x = torch.matmul(torch.linalg.pinv(self.channel_matrix), x)
        
        x = torch.cat((x.real, x.imag), dim=2)
        x = x.reshape(original_shape)

        return x

class MIMOCombiner(nn.Module):
    def __init__(self, n_streams, type="task-oriented") -> None:
        super().__init__()
        self.N_S = n_streams
        self.register_buffer('channel_matrix', None)
        self.type = type
        assert self.type in ["task-oriented", "SVD", "ZF", "LMMSE", "MCR2"]
        if self.type == "task-oriented":
            self.linear_weight1 = nn.Parameter(torch.eye(self.N_S * self.N_S, self.N_S * self.N_S, dtype=torch.complex64))
        
    def set_channel_matrix(self, channel_matrix):
        self.channel_matrix = channel_matrix
    
    def forward(self, x):
        # self.precoding_tensor = torch.view_as_complex(self.relu(torch.view_as_real(torch.matmul(torch.linalg.pinv(self.channel_matrix).flatten(1), self.linear_weight1.T))))
        # self.precoding_tensor = torch.view_as_complex(self.relu(torch.view_as_real(torch.matmul(self.precoding_tensor, self.linear_weight2.T)))).reshape(self.channel_matrix.shape)
        # Only linear layer without any activation function
        if self.type == "task-oriented":
            self.precoding_tensor = torch.matmul(torch.linalg.svd(self.channel_matrix)[0].conj().mT.flatten(1), self.linear_weight1.T).reshape(self.channel_matrix.shape)
            # self.precoding_tensor = self.precoding_tensor.unsqueeze(-1).unsqueeze(-1)
        elif self.type == "SVD":
            self.precoding_tensor = torch.linalg.svd(self.channel_matrix)[0].conj().mT
        elif self.type == "ZF":
            return x
        
        original_shape = x.shape
        x = x.reshape(x.shape[0], self.N_S, -1)
        x = torch.complex(*x.chunk(2, dim=2))
        # x = x.unsqueeze(0).unsqueeze(-1)
        
        x = torch.matmul(self.precoding_tensor, x)
        # x = F.conv2d(x, self.precoding_tensor)
        # x = x.squeeze(0).squeeze(-1)
        
        # x = torch.matmul(torch.linalg.pinv(self.channel_matrix), x)
        
        x = torch.cat((x.real, x.imag), dim=2)
        x = x.reshape(original_shape)

        return x
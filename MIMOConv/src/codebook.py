import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from psk import PSK
from mimo_comm import MIMOChannel
from codebook_output import CodebookOutput

from typing import Sequence

class Codebook(nn.Module):

    def __init__(self, latent_dim, n_embeddings, beta=1e-3, PSNR=20, n_streams=8):

        super().__init__()

        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        self.mod = PSK(M=n_embeddings)
        self.channel = MIMOChannel(n_streams=n_streams, PSNR=PSNR, model="rayleigh")
        self.beta = beta

        self.embedding = nn.Parameter(torch.Tensor(n_embeddings, latent_dim))
        nn.init.uniform_(self.embedding, -1/ n_embeddings, 1 / n_embeddings)
    
    def compute_score(self, x):
        return x @ self.embedding.T / math.sqrt(self.latent_dim)
        
    def send_over_channel(self, x):
        modulated = self.mod.modulate(x)
        received = self.channel(modulated)
        if self.channel.model == "rayleigh":
            equalized = (torch.linalg.lstsq(self.channel.channel_matrix, received.reshape(self.channel.N_S, -1)).solution).reshape(received.shape)
            demodulated = self.mod.demodulate(equalized)
        elif self.channel.model == "awgn":
            demodulated = self.mod.demodulate(received)
        return demodulated
    
    def construct_noise(self, samples):
        symbols = samples.argmax(dim=-1)
        symbols_distorted = self.send_over_channel(symbols)
        noise = F.one_hot(symbols_distorted, num_classes=self.n_embeddings).float() -\
            F.one_hot(symbols, num_classes=self.n_embeddings).float()
        return noise
    
    def sample(self, score):
        if self.training:
            samples = F.gumbel_softmax(score, tau=0.5, hard=True)
            noise = self.construct_noise(samples)
            samples = samples + noise
        else:
            samples = score.argmax(dim=-1)
            samples = self.send_over_channel(samples)
            samples = F.one_hot(samples, num_classes=self.n_embeddings).float()

        return samples

    def forward(self, x):
        original_codebook_outputs = x
        was_codebook = type(x) == CodebookOutput

        if was_codebook:
            x = x.original_tensor

        initial_shape = x.shape
        if len(initial_shape) > 2:
            x_reshaped = x.view(-1, self.latent_dim)
        score = self.compute_score(x_reshaped)
        dist = score.softmax(dim=-1)
        samples = self.sample(score)

        res = samples @ self.embedding

        if len(initial_shape) > 2:
            res = res.view(*initial_shape)

        if was_codebook:
            codebook_outputs = original_codebook_outputs.codebook_outputs
            codebook_outputs.append([res, dist, self])
            output = CodebookOutput(original_codebook_outputs.original_tensor, codebook_outputs)
            return output
        else:
            output = CodebookOutput(x, [[res, dist, self]])
            return output

    def __repr__(self):
        return f'Codebook(latent_dim={self.latent_dim}, n_embeddings={self.n_embeddings})'
    
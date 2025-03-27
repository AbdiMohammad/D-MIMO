import math
import torch
import torch.nn as nn

class PSK(nn.Module):

    def __init__(self, M):
        super().__init__()

        self.M = M

        sins = (torch.arange(M) * 2 * torch.pi / M).sin().unsqueeze(1)
        coss = (torch.arange(M) * 2 * torch.pi / M).cos().unsqueeze(1)
        constellation = torch.cat([coss, sins], dim=1)
        self.register_buffer('constellation', constellation)

    def modulate(self, z):
        return self.constellation[z]
    
    def demodulate(self, x):
        return (x @ self.constellation.T).argmax(dim=-1)


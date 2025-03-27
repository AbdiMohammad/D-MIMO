import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import tensorboard

class InverseApprox(nn.Module):
    def __init__(self, n_streams, n_carriers=1) -> None:
        super().__init__()
        self.N_S = n_streams
        self.linear_weight1 = nn.Parameter(torch.randn(self.N_S * self.N_S, self.N_S * self.N_S, dtype=torch.complex64))
        self.linear_weight2 = nn.Parameter(torch.randn(self.N_S * self.N_S, self.N_S * self.N_S, dtype=torch.complex64))
        self.relu = nn.ReLU()
            
    def forward(self, x):
        x = torch.view_as_complex(self.relu(torch.view_as_real(torch.matmul(x.view(x.shape[0], -1), self.linear_weight1.T))))
        x = torch.view_as_complex(self.relu(torch.view_as_real(torch.matmul(x.view(x.shape[0], -1), self.linear_weight2.T)))).reshape(x.shape[0], self.N_S, self.N_S)
        
        return x

if __name__ == "__main__":
    
    n_streams = 8
    device = 'cuda'
    epochs = 1000
    batch_size = 1024
    ckpt_path = "results/inverse_precode.pth"
    
    writer = tensorboard.SummaryWriter(log_dir="runs/" + "inverse_precode")
    
    model = InverseApprox(n_streams)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        inputs = torch.randn((batch_size, n_streams, n_streams), dtype=torch.complex64)
        inputs = inputs.to(device)
        
        targets = torch.linalg.pinv(inputs)
        targets = targets.to(targets)
        
        outputs = model(inputs)
        
        loss = criterion(torch.view_as_real(outputs), torch.view_as_real(targets))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar(f"Loss/train", loss.item() / batch_size, epoch)
        writer.flush()
    
    torch.save(model.state_dict(), ckpt_path)
        
import random
import numpy
import tqdm

import torch
from torch.utils import tensorboard, data
from pathlib import Path

from models.superwideisonet import *
from models.superwideresnet import *
from datasets import *
from mixup import *
import mimo_comm
from comm_utils import *

if __name__ == "__main__":
    random_seed = 1
    pretrained_path = "results/Experiment4_MIMONet-10_CIFAR10_HRR_4_10_4_None_False_False_False_0.1_0.0001_1.0_None_1024_0_1200_0.2_1e-05_1.0_10.0_1_model.pt"
    ckpt_path = "results/reconst_precode.pth"
    batch_size = 1024
    epochs = 500
    split_layer = "layer1"
    n_streams = 8
    
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # seed dataloader workers
    def seed_worker(worker_id):
        numpy.random.seed(random_seed)
        random.seed(random_seed)
    # seed generators
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    writer = tensorboard.SummaryWriter(log_dir='runs/' + 'reconst_precode')
    
    device = 'cuda'
    
    mimo_model = SuperWideISONet(num_img_sup_cap = 4, binding_type = "HRR", width=10, layers= [1, 1, 1], initial_width=4, num_classes=10, norm_layer=None, block=AdjustedISOBlock, dirac_init=False, relu_parameter=None, skip_init=False, trainable_keys=True, input_channels=3)
    
    pretrained_dict = torch.load(pretrained_path)["model_state_dict"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mimo_model.state_dict() and v.size() == mimo_model.state_dict()[k].size()}
    mimo_model.load_state_dict(pretrained_dict)
    for parameters in mimo_model.parameters():
        parameters.requires_grad = False
    exec(f"mimo_model.{split_layer} = StopCompute(mimo_model.{split_layer})")
    mimo_model = mimo_model.to(device)
    
    channel = mimo_comm.MIMOChannel(n_streams=n_streams, PSNR=20.0, model="noiseless-rayleigh")
    precoder = mimo_comm.MIMOPrecoder(n_streams=n_streams)
    precoder.set_channel_matrix(channel.channel_matrix)
    
    model = nn.Sequential(
        precoder,
        channel
    )
    model = model.to(device)
        
    trainset, evalset = get_train_eval_test_sets("CIFAR10", True)
    trainloader = data.DataLoader(trainset, batch_size=batch_size * 4, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
    evalloader = data.DataLoader(evalset, batch_size=batch_size * 4, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
        
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0.0
        total = 0
        for _ in range(4):
            for inputs, labels in trainloader:
                if inputs.shape[0] != batch_size * 4:
                    continue
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                inputs, _, _, _ = mixup_data(inputs, labels)
                
                try:
                    _ = mimo_model(inputs)
                except Exception as e:
                    latents = e.args[0].detach()
                    latents = latents.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(latents)
                
                loss = criterion(outputs, latents)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total += inputs.shape[0]
        
        if total != 0:
            writer.add_scalar(f"Loss/train", running_loss / total, epoch)
            writer.flush()
        if epoch % 10 == 0:
            model.get_submodule("1").update_channel_matrix()
            model.get_submodule("1").to(device)
            model.get_submodule("0").set_channel_matrix(model.get_submodule("1").channel_matrix)
            model.get_submodule("0").to(device)
    
    torch.save(model.get_submodule("0").state_dict(), ckpt_path)
                
                
                
                
                
                                
    
    
    
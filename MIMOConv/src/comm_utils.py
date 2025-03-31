import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn
import sklearn.cluster

from codebook import Codebook

import mimo_comm

class StopCompute(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    
    def forward(self, x):
        res = self.inner(x)
        raise Exception(res)

def model_device(model):
    return next(model.parameters()).device

def get_module_output_shape(model, dataloader, module):
    exec(f"model.{module} = StopCompute(model.{module})")

    try:
        _ = model(next(iter(dataloader))[0].to(model_device(model)))
    except Exception as e:
        return e.args[0].detach().shape
    finally:
        exec(f"model.{module} = model.{module}.inner")

def get_initial_weights(model, dataloader, module, latent_dim, n_embeddings):
    latent_list = []
    exec(f"model.{module} = StopCompute(model.{module})")

    for xs, labels in dataloader:
        xs = xs.to(model_device(model))
        labels = labels.to(model_device(model))
        try:
            _ = model(xs)
        except Exception as e:
            latent_list.append(e.args[0].detach())

    embeddings = torch.cat(latent_list).cpu().numpy().reshape(-1, latent_dim)
    k_means = sklearn.cluster.MiniBatchKMeans(n_clusters=n_embeddings)
    k_means.fit(embeddings)

    exec(f"model.{module} = model.{module}.inner")
    return k_means.cluster_centers_

def insert_codebook(model, dataloader, module, n_embeddings, beta, snr):
    latent_dim = get_module_output_shape(model, dataloader, module)[1]

    embeddings = get_initial_weights(model, dataloader, module, latent_dim, n_embeddings)

    target_module = eval(f"model.{module}")
    codebook = Codebook(latent_dim, n_embeddings, beta, PSNR=snr).to(model_device(model))
    codebook.embedding.data = torch.Tensor(embeddings).to(model_device(model))

    new_module = nn.Sequential(
        target_module,
        codebook
        )
    
    exec(f"model.{module} = new_module")

def insert_mimo_channel(model, split_layer, n_streams, channel_model, snr, precoder_type, batch_size):
    target_module = eval(f"model.{split_layer}")

    channel = mimo_comm.MIMOChannel(batch_size, n_streams=n_streams, model=channel_model, PSNR=snr).to(model_device(model))
    
    # latent_shape = get_module_output_shape(model, dataloader, split_layer)
    if precoder_type is not None:
        precoder = mimo_comm.MIMOPrecoder(n_streams=n_streams, type=precoder_type).to(model_device(model))
        precoder.set_channel_matrix(channel.channel_matrix)

        combiner = mimo_comm.MIMOCombiner(n_streams=n_streams, type=precoder_type).to(model_device(model))
        combiner.set_channel_matrix(channel.channel_matrix)
    else:
        precoder = nn.Identity()
        combiner = nn.Identity()
    
    # Loading the pinv equivalent weights
    # precoder.load_state_dict(torch.load("results/reconst_precode.pth"))
    # channel.set_channel_matrix(precoder.channel_matrix)

    new_module = nn.Sequential(
        target_module,
        # nn.Tanh(),
        precoder,
        channel,
        combiner
    )

    exec(f"model.{split_layer} = new_module")

    return model

# def insert_precoder()
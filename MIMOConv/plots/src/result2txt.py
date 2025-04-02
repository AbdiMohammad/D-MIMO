import random
import math
import torch

depths = [28, 16, 10]
widths = [10, 8, 4]

# Generate text result files for end-to-end latency
for model_idx in range(len(depths)):
    dataset = "CIFAR10"

    xlabels = ["DEC/MIMO", "BF-9/SISO", "BF-3/SISO", "DEC/MIMO", "BF-9/SISO", "BF-3/SISO", "DU-BCA-MM/SISO", "D-MIMO"]
    xlabels_hyper = ["LoS"] * 3 + ["NLoS"] * 3 + [None] + [None]

    types = ["HRR", None, None] * 2 + [None] + ["HRR"]
    nums = [4, 1, 1] * 2 + [1] + [4]
    initial_widths = [4, 1, 1] * 2 + [1] + [4]
    batch_sizes = [1024, 128, 128] * 2 + [128] + [1024]
    number_of_cpus = [0, 0, 0] * 2 + [0] + [16]
    experiment_nrs = [4, 0, 0] * 2 + [0] + [6]
    epochs = [1200, 200, 200] * 2 + [200] + [1200]
    bottlefit_sizes = [None, 9, 3] * 2 + [None] + [None]
    models = [f"MIMONet-{depths[model_idx]}", f"WideResNet-{depths[model_idx]}-BF-9", f"WideResNet-{depths[model_idx]}-BF-3"] * 2 + [f"WideResNet-{depths[model_idx]}"] + [f"MIMODistNet-{depths[model_idx]}"]

    result_txt_file = open(f"MIMOConv/plots/results_txt/latency/WRN-{depths[model_idx]}-{widths[model_idx]}.txt", "w")
    for idx in range(len(models)):
        result_path = f"MIMOConv/results/Experiment{experiment_nrs[idx]}_{models[idx]}_{dataset}_{types[idx]}_{nums[idx]}_{widths[model_idx]}_{initial_widths[idx]}_None_False_False_False_0.1_0.0001_1.0_None_{batch_sizes[idx]}_{number_of_cpus[idx]}_{epochs[idx]}_0.2_1e-05_1.0_10.0_1_measure.pt"
        result = torch.load(result_path)

        MD_time = result['MD']['Time'] / nums[idx]

        if xlabels_hyper[idx] == "NLoS":
            packet_loss = random.uniform(15.0, 25.0)
            COMM_time = result['COMM']['Time'] * (1 + (packet_loss / 100.0))
        else:
            COMM_time = result['COMM']['Time']
        COMM_time = COMM_time / nums[idx]

        ES_time = result['ES']['Time'] / nums[idx]

        result_txt_file.write(f"{MD_time:.4f}\t{COMM_time:.4f}\t{ES_time:.4f}\t# {xlabels[idx]}/{xlabels_hyper[idx]}\n")
    result_txt_file.close()

# Generate text result files for energy consumption
for model_idx in range(len(depths)):
    dataset = "CIFAR10"

    xlabels = ["DEC/MIMO", "BF-9/SISO", "BF-3/SISO", "DEC/MIMO", "BF-9/SISO", "BF-3/SISO", "DU-BCA-MM/SISO", "D-MIMO"]
    xlabels_hyper = ["LoS"] * 3 + ["NLoS"] * 3 + [None] + [None]

    types = ["HRR", None, None] * 2 + [None] + ["HRR"]
    nums = [4, 1, 1] * 2 + [1] + [4]
    initial_widths = [4, 1, 1] * 2 + [1] + [4]
    batch_sizes = [1024, 128, 128] * 2 + [128] + [1024]
    number_of_cpus = [0, 0, 0] * 2 + [0] + [16]
    experiment_nrs = [4, 0, 0] * 2 + [0] + [6]
    epochs = [1200, 200, 200] * 2 + [200] + [1200]
    bottlefit_sizes = [None, 9, 3] * 2 + [None] + [None]
    models = [f"MIMONet-{depths[model_idx]}", f"WideResNet-{depths[model_idx]}-BF-9", f"WideResNet-{depths[model_idx]}-BF-3"] * 2 + [f"WideResNet-{depths[model_idx]}"] + [f"MIMODistNet-{depths[model_idx]}"]

    result_txt_file = open(f"MIMOConv/plots/results_txt/energy/WRN-{depths[model_idx]}-{widths[model_idx]}.txt", "w")
    for idx in range(len(models)):
        result_path = f"MIMOConv/results/Experiment{experiment_nrs[idx]}_{models[idx]}_{dataset}_{types[idx]}_{nums[idx]}_{widths[model_idx]}_{initial_widths[idx]}_None_False_False_False_0.1_0.0001_1.0_None_{batch_sizes[idx]}_{number_of_cpus[idx]}_{epochs[idx]}_0.2_1e-05_1.0_10.0_1_measure.pt"
        result = torch.load(result_path)

        MD_energy = result['MD']['Power'] / nums[idx] * result['MD']['Time'] / nums[idx] / 1000.0 # From uJ to mJ

        if xlabels_hyper[idx] == "NLoS":
            packet_loss = random.uniform(15.0, 25.0)
            COMM_energy = result['COMM']['Energy'] * (1 + (packet_loss / 100.0)) * (1 + (packet_loss / 100.0))
        else:
            COMM_energy = result['COMM']['Energy']
        COMM_energy = COMM_energy / nums[idx] / nums[idx]
        
        result_txt_file.write(f"{MD_energy:.4f}\t{COMM_energy:.4f}\t# {xlabels[idx]}/{xlabels_hyper[idx]}\n")
    result_txt_file.close()

# Generate text result files for total transmitted I/Q symbols
for model_idx in range(len(depths)):
    dataset = "CIFAR10"

    xlabels = ["DEC/MIMO", "BF-9/SISO", "BF-3/SISO", "DEC/MIMO", "BF-9/SISO", "BF-3/SISO", "DU-BCA-MM/SISO", "D-MIMO"]
    xlabels_hyper = ["LoS"] * 3 + ["NLoS"] * 3 + [None] + [None]

    types = ["HRR", None, None] * 2 + [None] + ["HRR"]
    nums = [4, 1, 1] * 2 + [1] + [4]
    initial_widths = [4, 1, 1] * 2 + [1] + [4]
    batch_sizes = [1024, 128, 128] * 2 + [128] + [1024]
    number_of_cpus = [0, 0, 0] * 2 + [0] + [16]
    experiment_nrs = [4, 0, 0] * 2 + [0] + [6]
    epochs = [1200, 200, 200] * 2 + [200] + [1200]
    bottlefit_sizes = [None, 9, 3] * 2 + [None] + [None]
    models = [f"MIMONet-{depths[model_idx]}", f"WideResNet-{depths[model_idx]}-BF-9", f"WideResNet-{depths[model_idx]}-BF-3"] * 2 + [f"WideResNet-{depths[model_idx]}"] + [f"MIMODistNet-{depths[model_idx]}"]

    result_txt_file = open(f"MIMOConv/plots/results_txt/symbols/WRN-{depths[model_idx]}-{widths[model_idx]}.txt", "w")
    for idx in range(len(models)):
        result_path = f"MIMOConv/results/Experiment{experiment_nrs[idx]}_{models[idx]}_{dataset}_{types[idx]}_{nums[idx]}_{widths[model_idx]}_{initial_widths[idx]}_None_False_False_False_0.1_0.0001_1.0_None_{batch_sizes[idx]}_{number_of_cpus[idx]}_{epochs[idx]}_0.2_1e-05_1.0_10.0_1_measure.pt"
        result = torch.load(result_path)

        if xlabels_hyper[idx] == "NLoS":
            packet_loss = random.uniform(15.0, 25.0)
            COMM_symbols = math.ceil(result['COMM']['Symbols'] * (1 + (packet_loss / 100.0)))
        else:
            COMM_symbols = result['COMM']['Symbols']
        COMM_symbols = math.ceil(COMM_symbols / nums[idx])

        result_txt_file.write(f"{COMM_symbols}\t# {xlabels[idx]}/{xlabels_hyper[idx]}\n")
    result_txt_file.close()
#! /bin/bash

EXEC_DEVICE=$1
GPU_DEVICE=$2

depths=(28 16 10)
widths=(10 8 4)

dataset=CIFAR10
batch_size_test=64

type=HRR
num=4
initial_width=4

batch_size_train=1024
number_of_cpus_train=0
number_of_cpus_test=6

experiment_nr=4

epochs=1200

# Get the measurements on Jetson Orin Nano
cd ~/MAbdi/D-MIMO/MIMOConv/
for model_idx in "${!depths[@]}"
do
    depth=${depths[model_idx]}
    width=${widths[model_idx]}
    model=MIMONet-${depth}
    model_name=${model}-${width}
    echo ${model_name}

    checkpoint=results/Experiment${experiment_nr}_${model}_${dataset}_${type}_${num}_${width}_${initial_width}_None_False_False_False_0.1_0.0001_1.0_None_${batch_size_train}_${number_of_cpus_train}_${epochs}_0.2_1e-05_1.0_10.0_1_model.pt
    CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python src/Test_MIMODistNet.py ${model} ${dataset} ${type} ${num} --width ${width} --initial_width ${initial_width} --batch_size ${batch_size_test} --number_of_cpus ${number_of_cpus_test} --checkpoint ${checkpoint} --split_layer layer1 --partition ${EXEC_DEVICE}
done

# ssh jetson << EOF
#     cd ~/MAbdi/D-MIMO/MIMOConv/
#     for model_idx in "${!depths[@]}"
#     do
#         depth=${depths[model_idx]}
#         width=${widths[model_idx]}
#         model=MIMONet-\${depth}
#         model_name=\${model}-\${width}
#         echo \${model_name}

#         checkpoint=results/Experiment${experiment_nr}_\${model}_${dataset}_${type}_${num}_\${width}_${initial_width}_None_False_False_False_0.1_0.0001_1.0_None_${batch_size_train}_${number_of_cpus_train}_${epochs}_0.2_1e-05_1.0_10.0_1_model.pt
#         CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python src/Test_MIMODistNet.py \${model} ${dataset} ${type} ${num} --width \${width} --initial_width ${initial_width} --batch_size ${batch_size_test} --number_of_cpus ${number_of_cpus_test} --checkpoint \${checkpoint} --split_layer layer1 --channel_model rayleigh --precoder_type task-oriented --partition ${EXEC_DEVICE}
#     done
# EOF
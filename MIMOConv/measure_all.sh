#! /bin/bash

GPU_DEVICE=$1

ssh jetson "rm -rf /home/mabdi/MAbdi/D-MIMO/MIMOConv/results"

scp -r /home/microway/MAbdi/multiple-input-multiple-output-nets/.bkup/results jetson:/home/mabdi/MAbdi/D-MIMO/MIMOConv/

ssh -tt jetson << EOF
    cd ~/MAbdi/D-MIMO/MIMOConv/
    chmod +x ./measure_DMIMO.sh
    chmod +x ./measure_MIMONet.sh
    chmod +x ./measure_BF.sh
    chmod +x ./measure_WideResNet.sh
    ./measure_DMIMO.sh ${GPU_DEVICE}
    ./measure_MIMONet.sh ${GPU_DEVICE}
    ./measure_BF.sh ${GPU_DEVICE}
    ./measure_WideResNet.sh ${GPU_DEVICE}
EOF
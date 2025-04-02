#! /bin/bash

EXEC_DEVICE=$1
GPU_DEVICE=$2

cd ~/MAbdi/D-MIMO/MIMOConv/
chmod +x scripts/measure_MIMONet.sh
chmod +x scripts/measure_BF.sh
chmod +x scripts/measure_DMIMO.sh
chmod +x scripts/measure_WideResNet.sh
scripts/measure_MIMONet.sh ${EXEC_DEVICE} ${GPU_DEVICE}
scripts/measure_BF.sh ${EXEC_DEVICE} ${GPU_DEVICE}
scripts/measure_DMIMO.sh ${EXEC_DEVICE} ${GPU_DEVICE}
scripts/measure_WideResNet.sh ${EXEC_DEVICE} ${GPU_DEVICE}
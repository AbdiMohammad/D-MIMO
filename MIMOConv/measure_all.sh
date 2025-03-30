#! /bin/bash

GPU_DEVICE=$1

chmod +x ./measure_DMIMO.sh
chmod +x ./measure_MIMONet.sh
chmod +x ./measure_BF.sh
chmod +x ./measure_WideResNet.sh

./measure_DMIMO.sh ${GPU_DEVICE}
./measure_MIMONet.sh ${GPU_DEVICE}
./measure_BF.sh ${GPU_DEVICE}
./measure_WideResNet.sh ${GPU_DEVICE}

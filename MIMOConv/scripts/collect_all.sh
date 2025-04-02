#! /bin/bash

GPU_DEVICE=$1

# ssh jetson "rm -rf /home/mabdi/MAbdi/D-MIMO/MIMOConv/results/*"

# scp -r /home/microway/MAbdi/D-MIMO/.bkup/results/*_model.pt jetson:/home/mabdi/MAbdi/D-MIMO/MIMOConv/results/

# ssh jetson "/home/mabdi/MAbdi/D-MIMO/MIMOConv/scripts/measure_all.sh MD ${GPU_DEVICE}"

scp -r jetson:/home/mabdi/MAbdi/D-MIMO/MIMOConv/results/* /home/microway/MAbdi/D-MIMO/MIMOConv/results/

/home/microway/MAbdi/D-MIMO/MIMOConv/scripts/measure_all.sh ES ${GPU_DEVICE}

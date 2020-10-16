#!/bin/bash -e

JOB_NAME="sparse_sensing" \
DOCKER_IMAGE_NAME="dtr.thefacebook.com/minhpvo/sparse-sensing:v33" \
INIT_SCRIPT="dgx/my_init.sh" \
CLEANUP_SCRIPT="dgx/my_cleanup.sh" \
NUM_NODES=1 \
GPUS_PER_TASK=8 \
CPUS_PER_TASK=40 \
MEM_PER_CPU=5g \
EXTRA_DOCKER_ARGS="--ipc=host -v /mnt/surreal_ssd:/mnt/surreal_ssd" \
EXTRA_SLURM_CMDS="-x sea104-dgx112" \
$(pwd -P)/../tools/run_slurm.sh $(pwd -P)/dgx/run_pipeline5_rgbspec_san19pairfreeze_actreggru2_top1.sh

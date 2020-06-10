#/usr/bin/env bash
#conda activate sparse_sensing

# Get GPUID
if [ -z $1 ]; then
    GPUID=-1
else
    GPUID=$1
fi

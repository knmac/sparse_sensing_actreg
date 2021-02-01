#!/usr/bin/env bash
IN_DIR="./data/ActivityNet/ActivityNet-v1.3"
OUT_DIR="./data/ActivityNet/frames_256"
mkdir -p $OUT_DIR

for pth in $IN_DIR/*; do
    vid="${pth##*/}"
    vid="${vid%.*}"
    
    output="$OUT_DIR/$vid"
    mkdir -p $output
    # Extract if the output dir is empty -> allow resuming
    if [ -z "$(ls -A $output)" ]; then
        echo $vid
        ffmpeg \
            -i "$pth" \
            -vf 'scale=-2:256' \
            -q:v 4 \
            "$output/%4d.jpg"
    fi
done

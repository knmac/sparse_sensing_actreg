#!/usr/bin/env bash
# Batch extract frames for all videos in the datasets. Need to execute
# `run_ffmpeg_docker.sh` first. Modify IN_DIR and OUT_DIR if needed.

IN_DIR='/data/EPIC_KITCHENS_2018/videos'
OUT_DIR='/data/EPIC_KITCHENS_2018/frames_untar/rgb'
mkdir -p $OUT_DIR 

# Extract train videos
for vid in $IN_DIR/train/*/*; do
    output="${vid//$IN_DIR/$OUT_DIR}"
    output="${output//\.MP4/}"
    mkdir -p $output

    ffmpeg \
        -hwaccel cuvid \
        -c:v "h264_cuvid" \
        -i "$vid" \
        -vf 'scale_npp=-2:256,hwdownload,format=nv12' \
        -q:v 4 \
        -r 60 \
        "$output/frame_%010d.jpg"
done

# Extract test videos
for vid in $IN_DIR/test/*/*; do
    output="${vid//$IN_DIR/$OUT_DIR}"
    output="${output//\.MP4/}"
    mkdir -p $output

    ffmpeg \
        -hwaccel cuvid \
        -c:v "h264_cuvid" \
        -i "$vid" \
        -vf 'scale_npp=-2:256,hwdownload,format=nv12' \
        -q:v 4 \
        -r 60 \
        "$output/frame_%010d.jpg"
done

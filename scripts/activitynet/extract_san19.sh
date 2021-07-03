#!/usr/bin/env bash
# Extract weight from SAN19 baselines

# 224x224
echo "extracting 224x224..."
python tools/extract_san_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/activitynet_run_pipeline_simple_san19pair_rgb_224/best.model" \
    --output_dir    "pretrained/activitynet/san19_224"


# 112x112
echo "extracting 112x112..."
python tools/extract_san_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/activitynet_run_pipeline_simple_san19pair_rgb_112/best.model" \
    --output_dir    "pretrained/activitynet/san19_112"

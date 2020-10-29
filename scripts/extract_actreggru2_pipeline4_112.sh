#!/usr/bin/env bash
# Extract weight from action recognition weight of pipeline 4 <=> top_0
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/pipeline4_rgbspec_san19pairfreeze112_actreggru2/best.model" \
    --output_dir    "pretrained/actreggru2_pipeline4_112/"

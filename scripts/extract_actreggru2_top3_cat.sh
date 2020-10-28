#!/usr/bin/env bash
# Extract weight from action recognition weight with top3_cat
python tools/extract_actreg_from_pipeline.py \
    --weight        "/home/knmac/Dropbox/SparseSensing/DGX_training_logs/pipeline5_rgbspec_san19pairfreeze_actreggru2_top1_cat/best.model" \
    --output_dir    "pretrained/actreggru2_top3_cat/"

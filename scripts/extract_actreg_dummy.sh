#!/usr/bin/env bash
# Extract weight from action recognition weight with top1_cat
# Dummy example
python tools/extract_actreg_from_pipeline.py \
    --weight        "saved_models/epic_kitchens_Pipeline5_RGBSpec_segs2_ep10_lr0.01_lr_st30_60_90_san19pairfreeze_actreggru2_top1_cat/Oct24_12-43-34/best.model" \
    --output_dir    "pretrained/actreggru2_top1_cat/"

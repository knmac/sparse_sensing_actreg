#!/usr/bin/env bash
ROOT="/home/knmac/Dropbox/SparseSensing/DGX_training_logs/finetune_pipeline4"
for freeze in "_freeze" ""; do
    for lr in 0.003 0.001 0.0003 0.0001; do
        for drop in 0.5 0.25 0; do
            echo "============================================================"
            echo $freeze $lr $drop
            dir="${ROOT}/epic_kitchens_Pipeline4_RGBSpec_segs10_ep50_lr${lr}_lr_st20_40_finetune112${freeze}_lr${lr}_drop${drop}"
            python tools/parse_best_tensorboard.py -d $dir/*

            echo
            echo
        done
    done
done

#!/usr/bin/env bash
IN_DIM="nofc1"
HID_DIM="1024hid"
N_LAY="2lay"

PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         "scripts/finetune_pipeline5/configs/pipeline5_top1_cat__${IN_DIM}_${HID_DIM}_${N_LAY}.yaml" \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'scripts/finetune_pipeline5/configs/train_san_freeze_adam_100_lr0.003.yaml' \
    --experiment_suffix "san19pairfreeze_actreggru2_top1_cat_${IN_DIM}_${HID_DIM}_${N_LAY}_lr0.003" \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'
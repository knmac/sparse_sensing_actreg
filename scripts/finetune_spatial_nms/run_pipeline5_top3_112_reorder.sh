#!/usr/bin/env bash
TOP="top3"
BBOX="112"
REORDER="reorder"


PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         "scripts/finetune_spatial_nms/configs/pipeline5_rgbspec_san19pairfreeze_actreggru3_${TOP}_${BBOX}_${REORDER}.yaml" \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_30_lr0.0003.yaml' \
    --experiment_suffix "finetune_nms__${TOP}_${BBOX}_${REORDER}" \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'

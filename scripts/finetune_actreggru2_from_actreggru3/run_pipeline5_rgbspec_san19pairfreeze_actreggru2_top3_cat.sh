#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'scripts/finetune_actreggru2_from_actreggru3/configs/pipeline5_rgbspec_san19pairfreeze_actreggru2_top3_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_30_lr0.0003.yaml' \
    --experiment_suffix 'san19pairfreeze_actreggru2_top3_cat__fromactreggru3' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'

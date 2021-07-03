#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/activitynet/pipeline5_rgb_san19pairfreeze_actreggru2_top2_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/activitynet.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_50.yaml' \
    --seed              1007 \
    --experiment_suffix 'san19pairfreeze_actreggru2_top2_cat' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'

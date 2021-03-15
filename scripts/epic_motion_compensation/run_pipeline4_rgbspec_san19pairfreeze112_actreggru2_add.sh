#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline4_rgbspec_san19pairfreeze112_actreggru2_add.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_compensation.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_50.yaml' \
    --experiment_suffix 'san19pairfreeze112_actreggru2_add_compensation' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'

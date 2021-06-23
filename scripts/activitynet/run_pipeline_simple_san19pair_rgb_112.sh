#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/activitynet/pipeline_simple_san19pair_rgb_112.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/activitynet.yaml' \
    --train_cfg   'configs/train_cfgs/activitynet/train_san50.yaml' \
    --seed        1007 \
    --experiment_suffix 'san19pair_112' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      'logs' \
    --savedir     'saved_models'

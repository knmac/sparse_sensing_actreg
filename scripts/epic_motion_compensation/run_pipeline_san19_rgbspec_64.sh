#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_san19pair_rgbspec_64_epicpretrained.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens_compensation.yaml' \
    --train_cfg   'configs/train_cfgs/train_san_50.yaml' \
    --experiment_suffix 'san19pair_64_avg' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      'logs' \
    --savedir     'saved_models'

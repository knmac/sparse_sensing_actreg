#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_san19pair_rgbdsspec_224.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens_rgbds.yaml' \
    --train_cfg   'configs/train_cfgs/train_san.yaml' \
    --experiment_suffix 'san19pair_224_avg' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      'logs' \
    --savedir     'saved_models'

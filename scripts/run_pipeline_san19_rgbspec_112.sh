#!/usr/bin/env bash
python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_san19pair_rgbspec_112.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg   'configs/train_cfgs/train_tbn.yaml' \
    --experiment_suffix 'san19pair_avg' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      'logs' \
    --savedir     'saved_models'

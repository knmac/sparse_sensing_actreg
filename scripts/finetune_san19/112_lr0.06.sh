#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_san19pair_rgbspec_112.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg   'scripts/finetune_san19/configs/train_san_lr0.06.yaml' \
    --experiment_suffix 'san19pair_112_avg_finetune_lr0.06' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      'logs' \
    --savedir     'saved_models'

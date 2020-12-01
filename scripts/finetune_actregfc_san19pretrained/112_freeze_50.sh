#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline_simple_san19pair_rgbspec_112_epicpretrained.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg         'scripts/finetune_actregfc_san19pretrained/configs/freeze_adam_50.yaml' \
    --experiment_suffix '112_freeze_50' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'

#!/usr/bin/env bash
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline6_rgbspec_san19pairfreeze_halluconv2_actreggru2_top0_cat.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'configs/train_cfgs/test_pipeline6.yaml' \
    --experiment_suffix 'san19pairfreeze_halluconv2_actreggru2_top0_cat' \
    --is_training       false \
    --logdir            'logs' \
    --savedir           'saved_models'

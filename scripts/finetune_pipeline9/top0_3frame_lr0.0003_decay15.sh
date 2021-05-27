#!/usr/bin/env bash
FRAME="3frame"
LR="lr0.0003_decay15"

PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         "scripts/finetune_pipeline9/configs/pipeline9_rgbspec_san19pairfreeze_halluconv2_actreggru2_timernn_top0_cat_${FRAME}.yaml" \
    --dataset_cfg       "configs/dataset_cfgs/epickitchens_noshuffle.yaml" \
    --train_cfg         "scripts/finetune_pipeline9/configs/train_san_freeze_adam_30_${LR}.yaml" \
    --experiment_suffix "finetune_pipeline9__${FRAME}_${LR}" \
    --is_training       true \
    --train_mode        "from_scratch" \
    --logdir            "logs" \
    --savedir           "saved_models"

#!/usr/bin/env bash
TOP="top2"
LR="lr0.001"
IN_DIM="nofc1"
HID_DIM="1024hid"
N_LAY="2lay"
EFF_WEIGHT="eff0.01"


PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         "scripts/finetune_pipeline8/configs/pipeline8__${TOP}_${EFF_WEIGHT}_${IN_DIM}_${HID_DIM}_${N_LAY}.yaml" \
    --dataset_cfg       "configs/dataset_cfgs/epickitchens_noshuffle.yaml" \
    --train_cfg         "scripts/finetune_pipeline8/configs/train_san_freeze_adam_50_${LR}.yaml" \
    --experiment_suffix "finetunepipeline8_${LR}_${TOP}_${EFF_WEIGHT}_${IN_DIM}_${HID_DIM}_${N_LAY}" \
    --is_training       true \
    --train_mode        "from_scratch" \
    --logdir            "logs" \
    --savedir           "saved_models"

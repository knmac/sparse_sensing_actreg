#!/usr/bin/env bash
LR="0.001"
DROP="0.25"

echo "========================================================================"
echo "LR=${LR}"
echo "DROP=${DROP}"
echo "========================================================================"

PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         "scripts/finetune_pipeline4_cat/configs_finetune/pipeline4_112_drop${DROP}.yaml" \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         "scripts/finetune_pipeline4_cat/configs_finetune/train_lr${LR}.yaml" \
    --experiment_suffix "finetune112_lr${LR}_drop${DROP}" \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            'logs' \
    --savedir           'saved_models'

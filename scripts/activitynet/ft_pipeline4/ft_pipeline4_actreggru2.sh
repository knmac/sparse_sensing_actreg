#!/usr/bin/env bash
N_DIMS=('_2048' '_1024' '_512', '_256')
N_LAYS=('_2lay' '_1lay')
LR=('_' '_lr0.0003' '_lr0.0001')

#N_DIMS=('_256')
#N_LAYS=('_2lay' '_1lay')
#LR=('_')

for dim in ${N_DIMS[@]}; do
    for lay in ${N_LAYS[@]}; do
        for lr in ${LR[@]}; do
            if [[ $lr == "_" ]]; then lr=""; fi

            PYTHONFAULTHANDLER=1 python main.py \
                --model_cfg         "scripts/activitynet/ft_pipeline4/cfg/pipeline4_112_actreggru2${dim}${lay}.yaml" \
                --dataset_cfg       "configs/dataset_cfgs/activitynet_short.yaml" \
                --train_cfg         "configs/train_cfgs/train_san_freeze_adam_50${lr}.yaml" \
                --seed              1007 \
                --experiment_suffix "ft_pipeline4_actreggru2_${dim}${lay}${lr}" \
                --is_training       true \
                --train_mode        "from_scratch" \
                --logdir            "logs" \
                --savedir           "saved_models"
        done
    done
done

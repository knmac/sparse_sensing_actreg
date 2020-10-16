#!/bin/bash -e
echo "BALLISTA_OUTDIR = $BALLISTA_OUTDIR"

BASEDIR=$(dirname "$0")
echo "BASEDIR = $BASEDIR"
cd "$BASEDIR/.."

export CUPY_CACHE_DIR="$BASEDIR/../.cupy/kernel_cache"
mkdir -p $CUPY_CACHE_DIR
echo "CUPY_CACHE_DIR = $CUPY_CACHE_DIR"


PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         'configs/model_cfgs/pipeline5_rgbspec_san19pairfreeze_actreggru2_top3.yaml' \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         'configs/train_cfgs/train_san_freeze_adam_100.yaml' \
    --experiment_suffix 'san19pairfreeze_actreggru2_top3' \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            $BALLISTA_OUTDIR'/logs' \
    --savedir           $BALLISTA_OUTDIR'/saved_models'

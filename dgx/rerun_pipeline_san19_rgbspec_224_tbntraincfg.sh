#!/bin/bash -e
echo "BALLISTA_OUTDIR = $BALLISTA_OUTDIR"

BASEDIR=$(dirname "$0")
echo "BASEDIR = $BASEDIR"
cd "$BASEDIR/.."

export CUPY_CACHE_DIR="$BASEDIR/../.cupy/kernel_cache"
mkdir -p $CUPY_CACHE_DIR
echo "CUPY_CACHE_DIR = $CUPY_CACHE_DIR"


PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_san19pair_rgbspec.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg   'configs/train_cfgs/train_tbn.yaml' \
    --experiment_suffix 'san19pair_224_avg_rerun_tbntraincfg' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      $BALLISTA_OUTDIR'/logs' \
    --savedir     $BALLISTA_OUTDIR'/saved_models'

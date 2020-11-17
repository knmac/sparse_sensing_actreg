#!/usr/bin/env bash
echo "BALLISTA_OUTDIR = $BALLISTA_OUTDIR"

BASEDIR=$(dirname "$0")
echo "BASEDIR = $BASEDIR"
cd "$BASEDIR/.."

export CUPY_CACHE_DIR="$BASEDIR/../.cupy/kernel_cache"
mkdir -p $CUPY_CACHE_DIR
echo "CUPY_CACHE_DIR = $CUPY_CACHE_DIR"


# copied from run_pipeline_rgbspec.sh
PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg   'configs/model_cfgs/pipeline_simple_rgbspec.yaml' \
    --dataset_cfg 'configs/dataset_cfgs/epickitchens.yaml' \
    --train_cfg   'configs/train_cfgs/train_tbn_nopartialbn.yaml' \
    --is_training true \
    --train_mode  'from_scratch' \
    --logdir      $BALLISTA_OUTDIR'/logs' \
    --savedir     $BALLISTA_OUTDIR'/saved_models'

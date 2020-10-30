#!/bin/bash -e
echo "BALLISTA_OUTDIR = $BALLISTA_OUTDIR"

BASEDIR=$(dirname "$0")
echo "BASEDIR = $BASEDIR"
cd "$BASEDIR/.."

export CUPY_CACHE_DIR="$BASEDIR/../.cupy/kernel_cache"
mkdir -p $CUPY_CACHE_DIR
echo "CUPY_CACHE_DIR = $CUPY_CACHE_DIR"

# -----------------------------------------------------------------------------

FREEZE=""
LR="0.003"
DROP="0.25"

echo "========================================================================"
echo "FREEZE=${FREEZE}"
echo "LR=${LR}"
echo "DROP=${DROP}"
echo "========================================================================"

PYTHONFAULTHANDLER=1 python main.py \
    --model_cfg         "scripts/finetune_pipeline4_cat/configs_finetune/pipeline4_112_drop${DROP}.yaml" \
    --dataset_cfg       'configs/dataset_cfgs/epickitchens_noshuffle.yaml' \
    --train_cfg         "scripts/finetune_pipeline4_cat/configs_finetune/train${FREEZE}_lr${LR}.yaml" \
    --experiment_suffix "finetune112${FREEZE}_lr${LR}_drop${DROP}" \
    --is_training       true \
    --train_mode        'from_scratch' \
    --logdir            $BALLISTA_OUTDIR'/logs' \
    --savedir           $BALLISTA_OUTDIR'/saved_models'

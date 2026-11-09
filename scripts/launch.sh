#!/bin/bash
# Run notebooks/02_train.ipynb headless in the background, for one group of models and training biomes.
#
# Usage:  bash launch.sh TAG MODELS TRAIN_BIOMES [SEEDS]
#   TAG           name for the log file and the executed notebook copy (logs/TAG.log, runs_nb/run_TAG.ipynb)
#   MODELS        comma-separated model keys, e.g. unet_ft,unet_pft,dinov2_ft  (see section 1 of the notebook)
#   TRAIN_BIOMES  amazon, atlantic or amazon,atlantic
#   SEEDS         default 0,1,2,3,4
#
# Several launches can share one GPU and one RESULTS_DIR: every run writes its own files (results/raw/*/<run>.csv), and runs
# that already have a result are skipped, so a crashed job is simply launched again.
#
# Environment (set below, change if your paths differ):
#   DATA_ROOT       extracted dataset (scripts/pod_setup.sh puts it in /root/data/bragagnolo)
#   RESULTS_DIR     where results/raw/ is written
#   DINOV3*_NAME    local folders with the DINOv3 weights in Hugging Face format. Leave them unset to download the official
#                   (gated) repositories instead: accept the DINOv3 license on Hugging Face and export HF_TOKEN.
set -euo pipefail
TAG=$1; MODELS=$2; BIOMES=$3; SEEDS=${4:-0,1,2,3,4}
WORK=${WORK:-/workspace}
cd "$WORK"; mkdir -p logs runs_nb
export DATA_ROOT=${DATA_ROOT:-/root/data/bragagnolo}
export RESULTS_DIR=${RESULTS_DIR:-$WORK/results}
export HF_HOME=${HF_HOME:-$WORK/hf}
export NUM_WORKERS=${NUM_WORKERS:-4} SAVE_CKPT=${SAVE_CKPT:-1}
[ -d "$WORK/weights/dinov3-vits16-pretrain-lvd1689m" ] && export DINOV3_NAME=${DINOV3_NAME:-$WORK/weights/dinov3-vits16-pretrain-lvd1689m}
[ -d "$WORK/weights/dinov3-vitl16-pretrain-lvd1689m-hf" ] && export DINOV3L_NAME=${DINOV3L_NAME:-$WORK/weights/dinov3-vitl16-pretrain-lvd1689m-hf}
[ -d "$WORK/weights/dinov3-vitl16-pretrain-sat493m-hf" ] && export DINOV3SAT_NAME=${DINOV3SAT_NAME:-$WORK/weights/dinov3-vitl16-pretrain-sat493m-hf}
MODELS=$MODELS TRAIN_BIOMES=$BIOMES SEEDS=$SEEDS setsid nohup jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 02_train.ipynb --output "runs_nb/run_$TAG.ipynb" > "logs/$TAG.log" 2>&1 &
echo "launched $TAG (pid $!): models=$MODELS biomes=$BIOMES seeds=$SEEDS"

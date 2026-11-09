#!/bin/bash
# Prepare a fresh GPU machine (we used RunPod, RTX 4090, image with PyTorch 2.14 + CUDA 13.0) for 02_train.ipynb.
#
# What it does, in order:
#   1. installs the Python packages the notebook needs, without replacing the image's PyTorch build;
#   2. downloads the dataset from Zenodo (record 4498086) and checks the MD5 sums published there;
#   3. extracts it with the official `unrar` (the archives use the RAR 7 format, which 7-Zip and older bsdtar cannot read);
#   4. copies the extracted images to the machine's local disk (reading 2,400 small files from a network volume is slow).
#
# Usage (on the GPU machine):  bash pod_setup.sh [WORK_DIR]       default WORK_DIR=/workspace
# Afterwards, start training with scripts/launch.sh.
set -euo pipefail
WORK="${1:-/workspace}"
LOCAL_DATA=/root/data/bragagnolo
cd "$WORK"

# 1. Python packages; the constraint file pins the preinstalled torch/torchvision versions
export PIP_BREAK_SYSTEM_PACKAGES=1
pip freeze | grep -iE '^(torch|torchvision)==' > constraints.txt || true
pip install -q -c constraints.txt rasterio segmentation-models-pytorch "transformers>=4.56" safetensors pillow pandas scipy \
    matplotlib nbconvert ipykernel

# 2. Dataset download (parallel connections) and MD5 check
command -v aria2c >/dev/null || (apt-get update -qq && apt-get install -y -qq aria2)
mkdir -p raw
URL=https://zenodo.org/api/records/4498086/files
aria2c -q -c -x16 -s16 -d raw -o AMAZON.rar "$URL/AMAZON.rar/content" &
aria2c -q -c -x16 -s16 -d raw -o ATLANTIC_FOREST.rar "$URL/ATLANTIC%20FOREST.rar/content" &
wait
echo "fd2a55b8437d65ac86e1a11161aeef98  raw/AMAZON.rar
86ac319b7ff6738a96f555abb79fb445  raw/ATLANTIC_FOREST.rar" | md5sum -c -

# 3. Extraction with RARLAB's unrar (freeware, Linux x64)
if [ ! -x rar/unrar ]; then
    curl -sfL -o rar.tgz https://www.rarlab.com/rar/rarlinux-x64-723.tar.gz
    tar --no-same-owner -xzf rar.tgz
fi
mkdir -p data && (cd data && ../rar/unrar x -o+ -idq ../raw/AMAZON.rar && ../rar/unrar x -o+ -idq ../raw/ATLANTIC_FOREST.rar)

# 4. Local copy for fast loading
mkdir -p "$LOCAL_DATA" && cp -r data/* "$LOCAL_DATA"/
echo "images: $(find "$LOCAL_DATA" -name '*.tif' | wc -l) files in $LOCAL_DATA"
echo SETUP_DONE

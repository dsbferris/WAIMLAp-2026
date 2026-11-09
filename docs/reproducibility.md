# Reproducibility

This document describes how the training runs were executed, how to repeat them, every script in `scripts/`, and every file
written to `results/`.

## 1. How the runs were executed

### Hardware and software

- Two RunPod Secure Cloud pods, each with one NVIDIA RTX 4090 (24 GB), at $0.89 per hour.
- RunPod image with PyTorch 2.14.1 and CUDA 13.0, Python 3.14.
- The other Python packages were installed by `scripts/pod_setup.sh` (with a constraint file, so the image's PyTorch build is kept).
  Every run records the GPU name and the PyTorch version in `results/raw/runs/<run>.csv` (columns `gpu`, `torch`).

### Data

`scripts/pod_setup.sh` downloads the two archives from Zenodo (record 4498086), checks their MD5 sums, extracts them with RARLAB
`unrar` 7.12 and copies the images to the pod's local disk (`/root/data/bragagnolo`). On a laptop, `notebooks/01_data.ipynb` does
the same (download, MD5 check, extraction) into `data/`.

| archive | MD5 |
|---|---|
| `AMAZON.rar` | `fd2a55b8437d65ac86e1a11161aeef98` |
| `ATLANTIC FOREST.rar` | `86ac319b7ff6738a96f555abb79fb445` |

The archives use the RAR 7 format (method v6). 7-Zip 26 and the `bsdtar` shipped with macOS cannot extract them. Use RARLAB
`unrar` 7.12 or newer (Linux) or `bsdtar` from Homebrew's `libarchive` (macOS).

### Pretrained weights

| model key | weights | source |
|---|---|---|
| `unet_*` | ResNet-34, ImageNet | downloaded by `segmentation_models_pytorch` |
| `dinov2_*` | `facebook/dinov2-small` | Hugging Face (Apache-2.0) |
| `dinov3_*` | `facebook/dinov3-vits16-pretrain-lvd1689m` | Hugging Face, gated (DINOv3 License) |
| `dinov3l_pft` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Hugging Face, gated (DINOv3 License) |
| `dinov3sat_pft` | `facebook/dinov3-vitl16-pretrain-sat493m` | Hugging Face, gated (DINOv3 License) |

For our runs, the DINOv3 weights were downloaded from Meta's release (`.pth` files; the download links are personal) and
converted to the Hugging Face format with `scripts/convert_dinov3_vit_to_hf.py`. This is the conversion script of `transformers`
(not in the pip package), with one added option, `--checkpoint`, to read a local `.pth` file:

```bash
uv run python scripts/convert_dinov3_vit_to_hf.py --model-name vits16_lvd1689m \
    --checkpoint dinov3_vits16_pretrain_lvd1689m-08c60483.pth --save-dir weights
uv run python scripts/convert_dinov3_vit_to_hf.py --model-name vitl16_lvd1689m \
    --checkpoint dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth --save-dir weights
uv run python scripts/convert_dinov3_vit_to_hf.py --model-name vitl16_sat493m \
    --checkpoint dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth --save-dir weights
```

The script checks the preprocessing and the first output values against reference numbers, so it downloads one COCO test image.
It writes `weights/<model-name>/`. Point `DINOV3_NAME`, `DINOV3L_NAME` and `DINOV3SAT_NAME` at these folders, or rename them to
the folder names that `scripts/launch.sh` looks for under `$WORK/weights/`. The converted `model.safetensors` files are SHA-256 identical to the official Hugging Face files:

| model | SHA-256 of `model.safetensors` |
|---|---|
| DINOv3 ViT-S/16, LVD-1689M | `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d` |
| DINOv3 ViT-L/16, LVD-1689M | `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` |
| DINOv3 ViT-L/16, SAT-493M | `4e6356d992c1301b5e7c275f465e47296c5c4ad17052e262b29fc21e82ccc698` |

So you can use the official Hugging Face files directly: accept the DINOv3 license on the model pages and set `HF_TOKEN`. Check a
download with `sha256sum model.safetensors`. We do not redistribute the weights.

### Parallel jobs

Each pod ran several `scripts/launch.sh` jobs at the same time on its single GPU, each for a different group of models and
training biomes. All jobs of a pod wrote to the same `RESULTS_DIR`. This is safe because every run writes its own files
(`results/raw/*/<run>.csv`), named by the run id `<model>__<train_biome>__s<seed>`, for example `dinov3_ft__amazon__s0`.

### Resuming

A run is complete when `results/raw/runs/<run>.csv` exists. This file is written last, after the per-image results, the learning
curve and the checkpoint. When `02_train.ipynb` starts, it skips every selected run that already has this file. A run that was
interrupted has no such file and is trained again from the start (there are no checkpoints in the middle of a run). So a crashed
or stopped job is simply launched again with the same arguments.

Runs are ordered seed by seed (all models and both training biomes for seed 0, then seed 1, ...), so an interrupted job leaves
complete seeds.

### Determinism

`02_train.ipynb` is set up so that the same seed on the same GPU type with the same library versions gives the same result:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set before CUDA starts), `torch.backends.cudnn.deterministic = True`,
  `torch.backends.cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Bilinear resizing inside the models is done as two matrix products (`Resize` class) instead of `F.interpolate`, whose backward pass
  is not deterministic on the GPU. The result of the forward pass is the same.
- Attention uses the plain math implementation (`sdpa_kernel(SDPBackend.MATH)`), not the fused kernels, and reduced-precision
  reductions in it are switched off.
- Python, NumPy and PyTorch are seeded with the run's seed. Data augmentation draws its random numbers from PyTorch; the
  `DataLoader` gets a generator seeded with the run's seed, which also seeds its worker processes.
- Mixed precision (autocast) is used on CUDA. It is deterministic, but its rounding depends on the GPU.

Different GPU types, CUDA or PyTorch versions give slightly different numbers. `warn_only=True` means that an operation without a
deterministic implementation would only print a warning; check the training log for such warnings.

### Python 3.14 notes

- `DataLoader` workers use `multiprocessing_context="fork"` (the default changed in Python 3.14).
- The RunPod image's Python is "externally managed"; `pod_setup.sh` sets `PIP_BREAK_SYSTEM_PACKAGES=1` for `pip install`.

## 2. Script reference

### `scripts/pod_setup.sh`

Prepares a fresh Linux GPU machine for `02_train.ipynb`.

```bash
bash pod_setup.sh [WORK_DIR]        # default WORK_DIR=/workspace
```

| argument | meaning |
|---|---|
| `WORK_DIR` | working folder; archives go to `WORK_DIR/raw`, extracted data to `WORK_DIR/data`, unrar to `WORK_DIR/rar` |

Steps:

1. installs `rasterio`, `segmentation-models-pytorch`, `transformers>=4.56`, `safetensors`, `pillow`, `pandas`, `scipy`,
   `matplotlib`, `nbconvert`, `ipykernel`, keeping the preinstalled `torch` and `torchvision` versions;
2. downloads both archives with `aria2c` (installed with `apt-get` if missing) and checks their MD5 sums;
3. downloads RARLAB `unrar` 7.12 for Linux x64 and extracts the archives;
4. copies the extracted data to `/root/data/bragagnolo` (local disk; reading many small files from a network volume is slow) and
   prints the number of `.tif` files. It prints `SETUP_DONE` at the end.

Environment: none required. Needs internet access, `pip`, `curl`, and `apt-get` if `aria2c` is not installed.

### `scripts/launch.sh`

Runs `02_train.ipynb` headless in the background (`jupyter nbconvert --execute`, detached with `setsid nohup`) for one group of
models and training biomes.

```bash
bash launch.sh TAG MODELS TRAIN_BIOMES [SEEDS]
# example
bash launch.sh vits unet_ft,dinov2_ft,dinov3_ft amazon,atlantic 0,1,2,3,4
```

| argument | meaning |
|---|---|
| `TAG` | name of the job; log in `WORK/logs/TAG.log`, executed notebook in `WORK/runs_nb/run_TAG.ipynb` |
| `MODELS` | comma-separated model keys: `unet_ft`, `unet_pft`, `dinov2_ft`, `dinov2_pft`, `dinov3_ft`, `dinov3_pft`, `dinov3l_pft`, `dinov3sat_pft` |
| `TRAIN_BIOMES` | `amazon`, `atlantic` or `amazon,atlantic` |
| `SEEDS` | comma-separated seeds, default `0,1,2,3,4` |

`02_train.ipynb` must be in `WORK`. Environment variables (defaults in brackets):

| variable | meaning |
|---|---|
| `WORK` [`/workspace`] | working folder |
| `DATA_ROOT` [`/root/data/bragagnolo`] | extracted dataset |
| `RESULTS_DIR` [`$WORK/results`] | where `raw/` is written |
| `HF_HOME` [`$WORK/hf`] | Hugging Face cache |
| `NUM_WORKERS` [`4`] | `DataLoader` worker processes |
| `SAVE_CKPT` [`1`] | `1` saves a checkpoint per run |
| `DINOV3_NAME`, `DINOV3L_NAME`, `DINOV3SAT_NAME` | local folders with DINOv3 weights in Hugging Face format. If unset and the folders `$WORK/weights/dinov3-vits16-pretrain-lvd1689m`, `$WORK/weights/dinov3-vitl16-pretrain-lvd1689m-hf` or `$WORK/weights/dinov3-vitl16-pretrain-sat493m-hf` exist, they are used; otherwise the official gated repositories are downloaded |
| `HF_TOKEN` | Hugging Face token, needed for the gated DINOv3 repositories |

Several launches can share one GPU and one `RESULTS_DIR` (see "Parallel jobs").

### Notebook settings (environment variables)

`01_data.ipynb`:

| variable | default | meaning |
|---|---|---|
| `DATA_ROOT` | `../data/bragagnolo` | extracted dataset; download and extraction are skipped if it is complete |
| `RAW_DIR` | `../data/raw` | downloaded archives |
| `RESULTS_DIR` | `../results` | output folder |

`02_train.ipynb`:

| variable | default | meaning |
|---|---|---|
| `DATA_ROOT` | `../data/bragagnolo` | extracted dataset |
| `RESULTS_DIR` | `../results` (`../results_smoke` with `SMOKE_TEST=1`) | output folder |
| `MODELS` | all 8 model keys | models to train |
| `SEEDS` | `0,1,2,3,4` | seeds |
| `TRAIN_BIOMES` | `amazon,atlantic` | training biomes |
| `EPOCHS` | `40` | epochs (one random 256 px crop per training image and epoch) |
| `NUM_WORKERS` | `4` | `DataLoader` worker processes |
| `SAVE_CKPT` | `1` | save a checkpoint per run |
| `DINOV2_NAME`, `DINOV3_NAME`, `DINOV3L_NAME`, `DINOV3SAT_NAME` | official Hugging Face ids | pretrained weights (id or local folder) |
| `HF_TOKEN` | unset | token for the gated DINOv3 repositories |
| `SMOKE_TEST` | `0` | `1`: tiny random models, 1 epoch, 2 checks; tests the pipeline |
| `MAX_IMAGES` | `0` | `>0`: use only that many images per split (for smoke tests) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | needed for deterministic matrix products on the GPU |

`04_qualitative.ipynb` uses the same variables as `02` (`DATA_ROOT`, `RESULTS_DIR`, `DINOV*_NAME`, `HF_TOKEN`) and executes sections 1–4 of `02_train.ipynb` (settings, data, models, prediction code; not section 5, which trains), so the model code exists once. It needs the checkpoints in `results/raw/ckpt/`. Extra variables:

| variable | default | meaning |
|---|---|---|
| `QUAL_SEED` | `0` | seed of the checkpoints to show |
| `QUAL_TRAIN_BIOME` | `amazon` | training biome; the figure shows the other biome |
| `QUAL_MODELS` | all models with a complete run | models (columns of the figure) |

It also checks that the loaded checkpoints reproduce the training-time confusion counts (on Apple MPS about 1e-4 of the pixels differ from the CUDA run, from floating-point differences).

## 3. Result files

| path | written by | content and columns |
|---|---|---|
| `results/data_manifest.csv` | `01_data.ipynb` | one row per image: `biome`, `split` (train/val/test), `file`, `satellite`, `date`, `orbit`, `tile` (Sentinel-2 tile), `patch_row`, `patch_col`, `crs`, `x_center`, `y_center` (patch centre in metres, UTM), `height`, `width`, `bands`, `image_dtype`, `mask_dtype`, `mask_values`, `same_georef` (image and mask have the same georeference), `zero_fraction` (share of pixels with all bands 0), `forest_fraction`, `mean_R`, `mean_G`, `mean_B`, `mean_NIR` (reflectance × 10 000), `ndvi_forest`, `ndvi_nonforest` (mean NDVI of mask-1 and mask-0 pixels) |
| `results/figures/fig0_examples.png` | `01_data.ipynb` | example test images of both biomes, RGB and RGB with forest mask |
| `results/figures/fig0_band_hist.png` | `01_data.ipynb` | reflectance distribution per band, Amazon vs. Atlantic Forest |
| `results/figures/fig0_ndvi.png` | `01_data.ipynb` | NDVI distribution of mask-1 and mask-0 pixels per biome |
| `results/raw/runs/<run>.csv` | `02_train.ipynb` | one row per run, written last (marks the run as complete): `run`, `model`, `seed`, `train_biome`, `epochs`, `steps`, `best_step` (step of the kept checkpoint), `val_miou` (best validation mIoU), `train_time_s`, `eval_time_s`, `total_params`, `trainable_params`, `gpu`, `torch`, `smoke_test` |
| `results/raw/per_image/<run>.csv` | `02_train.ipynb` | one row per evaluated image: `run`, `model`, `seed`, `train_biome`, `eval_set` (`amazon_test`, `atlantic_test`, or `<other biome>_all`), `biome`, `split`, `image` (file name without extension), `tp`, `fp`, `fn`, `tn` (pixel counts, forest = positive), `hist_nonforest_00`…`_19` and `hist_forest_00`…`_19` (pixel counts per bin of the predicted forest probability, 20 equal bins, for reference non-forest and forest pixels; used for AUROC, AP and calibration), `bnd_pred_matched`, `bnd_pred`, `bnd_true_matched`, `bnd_true` (boundary pixels of prediction and reference, and how many of them lie within 2 px of the other boundary; used for the boundary F1) |
| `results/raw/history/<run>.csv` | `02_train.ipynb` | learning curve, one row per validation check (15 per run): `step`, `epoch`, `lr`, `train_loss`, `train_miou`, `train_iou_forest`, `train_iou_nonforest`, `train_f1_forest`, `train_acc`, `train_precision`, `train_recall` (pooled over the augmented training crops since the previous check), `val_loss`, `val_miou`, `val_iou_forest`, `val_iou_nonforest`, `val_f1_forest`, `val_acc`, `val_precision`, `val_recall`, `elapsed_s` |
| `results/raw/ckpt/<run>.safetensors` | `02_train.ipynb` (`SAVE_CKPT=1`) | trained parameters and all buffers of the best checkpoint; frozen pretrained weights are not repeated. Not in git |
| `results/tables/*.csv` | `03_analysis.ipynb` | tables computed from `results/raw/` (for example `metrics_per_run.csv`, `table1_miou.csv`, ...); columns are described in that notebook |
| `results/figures/fig1_*.png`, `fig2_*.png`, ... | `03_analysis.ipynb` | result figures |
| `results/figures/fig4_qualitative_<train>_to_<other>_s<seed>.png` | `04_qualitative.ipynb` | example predictions on the other biome's test split (hard, typical and easy image = 10th, 50th and 90th percentile of the mean image mIoU over models) |
| `results/figures/fig_method.png` | `04_qualitative.ipynb` | method overview for the poster (example image per biome = test image with the median forest fraction) |
| `results/tables/qualitative_per_image_<train>_to_<other>_s<seed>.csv` | `04_qualitative.ipynb` | image mIoU per model behind that figure (a check, not a result; the paper's numbers come from `03`) |

The run id is `<model>__<train_biome>__s<seed>`. Every metric (IoU, mIoU, weighted IoU, per-image F1, ...) can be recomputed
from the confusion counts in `per_image/`.

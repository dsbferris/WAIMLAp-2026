# Forest segmentation across biomes: ImageNet U-Net vs. DINOv2 and DINOv3 encoders

Forest/non-forest segmentation of Sentinel-2 images is close to saturated inside one region: U-Nets trained and tested on the
Amazon reach an IoU of about 0.95, so this setting cannot tell encoders apart. Transfer to another biome does not saturate.
John & Zhang (2022) trained on the Amazon and reached a weighted IoU of only about 0.81 on the Atlantic Forest. We ask: **do
self-supervised foundation-model encoders transfer better between biomes than an ImageNet-pretrained ResNet-34 U-Net?** We compare
DINOv2 ViT-S/14, DINOv3 ViT-S/16, and DINOv3 ViT-L/16 pretrained either on web images (LVD-1689M) or on satellite images
(SAT-493M). All models use the same U-Net decoder and are trained with full (ft) or partial (pft) fine-tuning, with 5 seeds. Each
model is trained on one biome and evaluated on the official test sets of both biomes and on the whole other biome (605 images).

## Results

80 runs: 8 models × 2 training biomes × 5 seeds, one RTX 4090. mIoU, mean ± SD over seeds (`results/tables/table1_miou.csv`,
`table2_miou_whole_biome.csv`).

| Model | Amazon → Amazon | Amazon → Atlantic Forest | Amazon → whole Atlantic Forest (605) | Atlantic Forest → Atlantic Forest | Atlantic Forest → Amazon |
|---|---|---|---|---|---|
| U-Net ResNet-34, full | 0.964 ± 0.001 | 0.807 ± 0.009 | 0.814 ± 0.007 | 0.926 ± 0.002 | 0.918 ± 0.007 |
| U-Net ResNet-34, partial | 0.962 ± 0.002 | 0.814 ± 0.004 | 0.820 ± 0.003 | 0.929 ± 0.000 | 0.924 ± 0.006 |
| DINOv2 ViT-S/14, full | 0.960 ± 0.001 | 0.816 ± 0.003 | 0.823 ± 0.003 | 0.924 ± 0.001 | 0.933 ± 0.007 |
| DINOv2 ViT-S/14, partial | 0.958 ± 0.001 | 0.816 ± 0.007 | 0.824 ± 0.005 | 0.923 ± 0.001 | 0.938 ± 0.003 |
| DINOv3 ViT-S/16, full | 0.964 ± 0.001 | 0.816 ± 0.007 | 0.818 ± 0.007 | 0.931 ± 0.002 | 0.929 ± 0.005 |
| DINOv3 ViT-S/16, partial | 0.962 ± 0.002 | 0.811 ± 0.006 | 0.814 ± 0.005 | 0.930 ± 0.001 | 0.934 ± 0.006 |
| DINOv3 ViT-L/16 web, partial | 0.965 ± 0.000 | 0.819 ± 0.011 | 0.822 ± 0.013 | 0.932 ± 0.001 | 0.923 ± 0.004 |
| DINOv3 ViT-L/16 satellite, partial | 0.965 ± 0.001 | 0.802 ± 0.013 | 0.806 ± 0.010 | 0.933 ± 0.000 | 0.921 ± 0.011 |

- **In-biome results are saturated** (0.958–0.965 and 0.923–0.933 for all models).
- **Amazon → Atlantic Forest is a barrier for every model:** 0.80–0.82 mIoU (weighted IoU 0.82–0.84, John & Zhang 2022: ≈0.81).
  The errors are mostly false forest (whole Atlantic Forest: precision 0.81–0.85, recall 0.90–0.91). The reverse direction
  transfers much better.
- **No encoder is significantly better than the U-Net; the gains are at most ≈ 0.01 mIoU** (paired t-tests over seeds, Holm
  correction within each test set). Each model vs. the U-Net with full fine-tuning (`table3b_vs_unet_ft.csv`, 7 comparisons): the
  largest gains are +0.010 on the whole Atlantic Forest (DINOv2 ViT-S partial, Holm p = 0.31) and +0.012 on the official test
  split (DINOv3 ViT-L web, Holm p = 0.62); DINOv3 ViT-S full wins 5 of 5 seeds on the test split (+0.008, raw p = 0.04, Holm
  p = 0.29). The planned comparisons (`table3_paired.csv`) give the same picture. With 5 seeds, gains of about 0.01 cannot be
  ruled out.
- **Satellite pretraining does not help:** ViT-L satellite − web is −0.017 on the test split (Holm p = 0.57) and −0.016 on the
  whole biome; better in 1 of 5 seeds. SAT-493M is 0.6 m RGB imagery; our input is 10 m Sentinel-2 with NIR.
- A higher in-biome score does not predict better transfer (Spearman ρ = −0.31 against whole-biome mIoU over the 40
  Amazon-trained runs, −0.10 on the test split; `findings.txt`).
- No model needed a longer training schedule (`table6_convergence.csv`).

Figures: `results/figures/` (`fig1_in_vs_cross.png`, `fig2_in_vs_transfer_scatter.png`, `fig3_learning_curves_*.png`,
`fig4_qualitative_*.png`, `fig_method.png`). All metrics: `results/tables/`.

## Repository layout

```
notebooks/
  01_data.ipynb        download, extraction, checks and description of the dataset
  02_train.ipynb       training and evaluation; writes only raw CSV files (results/raw/)
  03_analysis.ipynb    tables, statistics and figures; reads only CSV files
  04_qualitative.ipynb example predictions for the poster/paper; needs the checkpoints (results/raw/ckpt/)
scripts/
  pod_setup.sh         prepares a fresh GPU machine (packages, dataset download and extraction)
  launch.sh            runs 02_train.ipynb headless in the background for a group of models
  convert_dinov3_vit_to_hf.py  converts Meta's DINOv3 .pth weights to Hugging Face format (from transformers, Apache-2.0)
results/
  data_manifest.csv    one row per image (written by 01)
  figures/             figures (fig0_* by 01, the others by 03)
  raw/                 runs/, per_image/, history/ CSV files (written by 02); ckpt/ is not in git
  checkpoints.csv      the 16 released checkpoints (seed, validation mIoU, SHA-256)
  tables/              tables (written by 03)
docs/
  reproducibility.md   how the runs were executed, script reference, list of result files and their columns
data/                  dataset (not in git; created by 01_data.ipynb or scripts/pod_setup.sh)
submission/            WAIMLAp 2026 poster and extended abstract (PDF)
course/                assignment and official templates (course material; not in the public repository)
```

## How to reproduce

### 1. Environment

We use [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync                      # creates .venv from pyproject.toml
```

or, with pip: `pip install -r requirements.txt`. On a GPU machine, install the PyTorch build that matches your CUDA version first.

### 2. Run the notebooks in order

```bash
cd notebooks
uv run jupyter nbconvert --to notebook --execute --inplace 01_data.ipynb      # data: download from Zenodo, checks, figures
uv run jupyter nbconvert --to notebook --execute --inplace 02_train.ipynb     # training: needs a GPU (see below)
uv run jupyter nbconvert --to notebook --execute --inplace 03_analysis.ipynb  # tables and figures from results/raw/
uv run jupyter nbconvert --to notebook --execute --inplace 04_qualitative.ipynb  # example predictions (needs checkpoints)
```

Paths and settings are read from environment variables (`DATA_ROOT`, `RESULTS_DIR`, `MODELS`, `SEEDS`, `TRAIN_BIOMES`, ...);
see `docs/reproducibility.md`. `SMOKE_TEST=1` runs 02 with tiny random models in a few minutes, to check that everything works.

### 3. GPU runs

We ran 02 on rented RunPod machines with one NVIDIA RTX 4090 each, using `scripts/pod_setup.sh` (packages and data) and
`scripts/launch.sh` (headless runs, several in parallel per GPU). Details are in `docs/reproducibility.md`.

Training is deterministic for a given GPU type: the same seed, GPU type and library versions give the same result. A different
GPU type or library version gives slightly different numbers.

The DINOv3 weights are gated on Hugging Face. Accept the DINOv3 license on the model pages and set `HF_TOKEN`, or point
`DINOV3_NAME`, `DINOV3L_NAME` and `DINOV3SAT_NAME` to local folders with the weights in Hugging Face format.

## Data and licenses

- **Code:** MIT License (see `LICENSE`).
- **Dataset:** Bragagnolo, da Silva & Grzybowski (2021), Zenodo, doi
  [10.5281/zenodo.4498086](https://doi.org/10.5281/zenodo.4498086), CC-BY-4.0. It is not included in this repository;
  `01_data.ipynb` downloads it. If you use it, cite it (see below). Note: the images are 16-bit reflectance (bands R, G, B, NIR), not
  8-bit as the Zenodo description says, and mask value 1 is forest (checked with NDVI in `01_data.ipynb`).
- **DINOv3 weights** (`facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-lvd1689m`,
  `facebook/dinov3-vitl16-pretrain-sat493m`): DINOv3 License. We do not redistribute them. Get them from Hugging Face after
  accepting the license.
- **DINOv2 weights** (`facebook/dinov2-small`): Apache-2.0, downloaded from Hugging Face at run time.
- **ResNet-34 ImageNet weights:** downloaded at run time by `segmentation_models_pytorch`.
- **Model checkpoints** hold only the trained parameters and buffers (the frozen encoder parts are the published pretrained
  weights). For each model and training biome, the seed with the best validation mIoU is attached to the GitHub release
  [`checkpoints-v1`](https://github.com/dsbferris/WAIMLAp-2026/releases/tag/checkpoints-v1) (16 files, about 2.6 GB;
  list and SHA-256 in `results/checkpoints.csv`). Put them in `results/raw/ckpt/` to run `04_qualitative.ipynb`. The `dinov3*`
  checkpoints derive from DINOv3 weights and fall under the DINOv3 License; `dinov2*` under Apache-2.0.

## References

1. Bragagnolo, L., da Silva, R. V., Grzybowski, J. M. V. (2021). *Amazon and Atlantic Forest image datasets for semantic
   segmentation*. Zenodo. https://doi.org/10.5281/zenodo.4498086
2. Bragagnolo, L., da Silva, R. V., Grzybowski, J. M. V. (2021). Amazon forest cover change mapping based on semantic
   segmentation by U-Nets. *Ecological Informatics*, 62, 101279. https://doi.org/10.1016/j.ecoinf.2021.101279
3. John, D., Zhang, C. (2022). An attention-based U-Net for detecting deforestation within satellite sensor imagery.
   *International Journal of Applied Earth Observation and Geoinformation*, 107, 102685. https://doi.org/10.1016/j.jag.2022.102685
4. Benvenuto et al. (2025). In: *Anais do Seminário Integrado de Software e Hardware (SEMISH 2025)*. SBC.
   https://doi.org/10.5753/semish.2025.8270 <!-- TODO: full author list and title -->
5. Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning robust visual features without supervision.
   arXiv:2304.07193.
6. Siméoni, O., Vo, H. V., Seitzer, M., et al. (2025). DINOv3. arXiv:2508.10104.
7. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In: *MICCAI
   2015*, LNCS 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

## AI assistance

The code and documentation were developed with Claude Code (Anthropic) as an assistant. The authors reviewed all of the work and
are responsible for it.

## Authors

C. S. Calderón Barrial, A. M. Vargas Zamora, E. Urbiola Ríos, F. Schwanebeck. Group INF659, Pontificia Universidad Católica del Perú (PUCP), semester 2026-2. <!-- TODO: check full names and accents -->

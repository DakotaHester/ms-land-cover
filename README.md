# MS Land Cover Project

End-to-end deep learning pipeline for multispectral land-cover mapping.

This repository contains the full workflow:
1. self-supervised pretraining of an encoder (`pretrain.py`),
2. supervised semantic segmentation fine-tuning (`finetune.py`),
3. tiled inference over large rasters (`inference.py`),
4. evaluation at model and raster-product levels (`test.py`, `assess.py`).

## Repository Mental Model

Primary executable scripts:
- `pretrain.py`: use BYOL/DINO/MoCov2 to learn encoder representations from unlabeled tiles.
- `finetune.py`: trains segmentation models on labeled split tiles.
- `inference.py`: applies trained model(s) to full-scene rasters using overlapping tiles.
- `test.py`: evaluates model checkpoint behavior on point/raster test inputs.
- `assess.py`: evaluates final predicted raster products against reference points.

Primary reusable package:
- `src/mslandcover/`
  - `models.py`: backbones, segmentation heads, SSL wrappers.
  - `loss.py`: supervised and SSL losses.
  - `metrics.py`: sklearn-based metric wrappers for training/evaluation loops.
  - `utils.py`: logging, device selection, checkpoint loading helpers.
  - `inference.py`: raster processing internals used by top-level inference script.
  - `data/`: datasets, transforms, and IO/statistics helpers.

## Class System

Land-cover class IDs and names are defined in `src/mslandcover/config.py` via `LEGEND_CLASSES`.
Class colors for raster outputs are also centralized there.

## Environment Setup

Recommended:

```bash
python -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -e .
```

Alternative pinned install:

```bash
pip install -r requirements.txt
```

Quick sanity checks:

```bash
python pretrain.py --help
python finetune.py --help
python inference.py --help
python test.py --help
python assess.py --help
```

## Fast End-To-End Smoke Test

1. Tiny pretraining run:

```bash
python pretrain.py \
  --pretrain_scheme byol \
  --pretrain_data_dir /path/to/pretrain \
  --pretrain_val_data_dir /path/to/pretrain_val \
  --num_epochs 300 \
  --mini_batch_size 16 \
  --full_batch_size 2048 \
```

2. One-fold fine-tuning run:

```bash
python finetune.py \
  --model unet \
  --encoder_weights /path/to/pretrained_encoder.pth \
  --split_dir ./data/splits \
  --n_train_samples 250 \
  --fold 1 \
  --num_epochs 1000 \
  --mini_batch_size 8 \
  --full_batch_size 8
```

3. One-raster inference:

```bash
python inference.py \
  --model unet \
  --input_dir /path/to/rasters \
  --output_dir /path/to/preds \
  --index 0 \
  --num_processes 1
```

4. Evaluation:

```bash
python test.py \
  --model unet \
  --model_weights /path/to/best_model.pth \
  --output_dir ./logs/test_debug
```

```bash
python assess.py \
  --prediction_raster /path/to/preds/example.tif \
  --ground_truth_shapefile /path/to/reference_points \
  --output_dir ./logs/assessment_debug
```

## Data Layouts (Inputs/Targets/Points)

This repo is very sensitive to *filesystem layout* and a few “silent” conventions.
If you match these, the scripts behave predictably.

### Band conventions (used everywhere)

Most training/inference code assumes a **3-band CIR composite** ordered as:

- Channel 0: **NIR**
- Channel 1: **Red**
- Channel 2: **Green**

Many NAIP products are 4-band (R, G, B, NIR). In several places the code will
convert 4-band imagery into CIR by selecting (NIR, R, G) = bands (4, 1, 2).

### Pretraining tiles layout (`pretrain.py`)

`pretrain.py` expects two folders:

- `--pretrain_data_dir`: directory of `*.tif` tiles
- `--pretrain_val_data_dir`: directory of `*.tif` tiles

There is no CSV manifest; files are discovered via `glob(<dir>/*.tif)`.

Practical expectations:

- Tiles are small enough to run as a batch (commonly `256×256`).
- Inputs should be 3-band CIR. If you feed 4-band NAIP tiles, the dataset
  converts to 3-band CIR (NIR, Red, Green).

### Fine-tuning splits layout (`finetune.py`)

`finetune.py` expects `--split_dir` with:

- a `splits.csv` file
- one folder per split column used by your selected row (e.g. `split_1/`, `split_2/`, ...)

Each split folder must contain:

- `input/*.tif` (imagery tiles)
- `target/*.tif` (label tiles)

Critical target encoding requirement:

- Target rasters must contain **class IDs 1..8** (not 0).
- The dataset converts targets to **0-based** indices internally via `target - 1`.
- If any target tile contains a `0`, `FineTuneDataset` raises an error.

### Point-based evaluation layout (`test.py`)

`test.py` evaluates a model against reference points by reading the *tile* that
contains each point and sampling the model output at that point pixel.

It expects:

- `--ground_truth_shapefile`: a vector dataset of points with at least an `id`
  field and a ground-truth class field.
- `--raster_dir`: directory of `*.tif` tiles.

Key filename contract:

- For a point with `id = XYZ`, the raster tile filename must be `XYZ.tif`.

### Product-level evaluation layout (`assess.py`)

`assess.py` evaluates a **completed prediction raster** (e.g. the output of
`inference.py`) against reference points.

It expects:

- `--prediction_raster`: single-band land-cover raster with class IDs `1..8` and
  nodata/background `0`.
- `--ground_truth_shapefile`: point dataset with ground-truth classes.

## Preprocessing (MrSID/NAIP → analysis-ready GeoTIFF)

Preprocessing utilities live in `src/mslandcover/data/preprocessing.py`.

The simple file-level pipeline is `preprocess_file(zip_path)`:

1. unzip `*.zip` into a folder next to the archive
2. convert `*.sid → *.tif` via `mrsiddecode`
3. resample + reproject to Mississippi Transverse Mercator via `gdalwarp`
   (default target resolution: 1m)
4. delete intermediate files

`preprocess_imagery.py` is an example driver that runs this over a directory of
zip archives using a thread pool.

External dependencies:

- `mrsiddecode` (for MrSID)
- `gdalwarp` / `gdalinfo` (GDAL)

If you already have GeoTIFFs, you can skip MrSID decode entirely.

## Normalization Statistics (mean/std)

Several scripts share the same normalization tensors:

- `./weights/pretrain_mean.pth`
- `./weights/pretrain_std.pth`

They can be produced in two ways:

1. Automatically during pretraining (`pretrain.py` computes them if missing).
2. Via the standalone helper `dataset_eda.py` (Dask-based mean/std computation).

If mean/std don’t match the imagery distribution you train/infer on, performance
will degrade (often dramatically).

## Self-Supervised Pretraining (`pretrain.py`): Deep Dive

`pretrain.py` learns an encoder using one of three objectives:

- **BYOL** (`--pretrain_scheme byol`)
- **MoCo v2** (`--pretrain_scheme moco`)
- **DINO** (`--pretrain_scheme dino`)

### What goes in

- Directory of `*.tif` tiles for training and validation.
- 3-band CIR is expected. If tiles are 4-band NAIP (R,G,B,NIR), the dataset
  converts them to CIR (NIR,R,G).

### How batching works (full batch via accumulation)

The effective batch size is:

$$\text{full\_batch\_size} = \text{mini\_batch\_size} \times \text{grad\_accum\_steps}$$

where `grad_accum_steps = full_batch_size // mini_batch_size`.

This is important for HPC runs: you can keep a large effective batch without
needing a huge GPU.

### Augmentations by scheme

- **BYOL**: two slightly different `BYOLDataAugmentation` transforms.
- **MoCo**: two `MoCoV2DataAugmentation` views.
- **DINO**: multi-crop `DINODataAugmentation` (2 global + N local crops).

### Outputs and artifacts

Run directories:

- Logs/checkpoints: `--log_dir/<encoder>/<scheme>/...` (defaults to a `resnet101` subdir)
- Exported weights: `--weights_dir/<encoder>/...`

Files written during a run:

- `log.txt` (full CLI config + progress)
- `history.csv` (train/val loss per epoch)
- `profiler.csv` (timing diagnostics)
- `model_complexity.json` (FLOPs/MACs/param counts)
- `checkpoint.pth` (resume-able full training state)
- `best_epoch.txt`
- `finished.txt`

Weights written to the weights directory:

- `<scheme>.pth`: best wrapper state when validation loss improves
- `<scheme>_last.pth`: exported **encoder** weights for fine-tuning
  - BYOL: online encoder
  - MoCo: query encoder
  - DINO: teacher encoder

### Resume and debug modes

- `--load_checkpoint` resumes from `checkpoint.pth` in the run log directory.
- `--debug` truncates datasets and reduces batch sizes for a quick smoke test.

## Supervised Fine-Tuning (`finetune.py`): Deep Dive

`finetune.py` trains a semantic segmentation model on labeled tiles.

### Split selection (`splits.csv`)

The script reads `splits.csv`, filters to the row matching:

- `n_train == --n_train_samples`
- `fold == --fold`

Then it treats any column whose value is `train` as a training split folder and
any column whose value is `val` as a validation split folder.

Example expectation:

- columns: `fold`, `n_train`, `split_1`, `split_2`, `split_3`, `split_4`, ...
- values: `train` / `val`

### Model architectures

Pick via `--model`:

- `unet`, `attention_unet`
- `deeplabv3plus`
- `upernet`, `pspnet`, `bisenet`, `danet`, `pan`, `fcn`
- `linear_probe` (simple head on top of the backbone)

### Encoder initialization (`--encoder_weights`)

- `--encoder_weights imagenet`: initialize backbone from torchvision ImageNet weights.
- `--encoder_weights /path/to/<scheme>_last.pth`: load a pretraining-exported encoder.

Under the hood, `adjust_backbone_weights()` remaps keys from various encoder
formats (BYOL/MoCo/DINO wrappers) into the segmentation backbone naming.

Current limitation:

- The CLI argument `--model_weights` is defined but not currently used.

### Training loop details

- Mixed precision via `autocast` + `GradScaler`.
- Gradient accumulation to reach `--full_batch_size`.
- Loss: `FocalLoss(gamma=--focal_gamma)`.
- Scheduler: `ReduceLROnPlateau` (stepped after warmup).
- Warmup: linear LR warmup for `--warmup_epochs`.
- Early stopping: stop if no validation-loss improvement for `--early_stopping_patience` epochs.

### Outputs and artifacts

Weights:

- `--output_dir/best_model.pth` when validation loss improves.

Logs (in `--log_dir`):

- `log.txt`
- `history.csv`
- `profiler.csv`
- `checkpoint.pth` (resume-able training state)
- `finished.txt`

Optional test-set evaluation (when `--test_dir` is provided):

- `confusion_matrix.csv`
- `classification_report.csv`
- `assessment_metrics.json`

## Evaluation (`test.py` vs `assess.py`)

There are two evaluation scripts with different purposes.

### `test.py`: model-level evaluation on labeled points

`test.py` loads a model checkpoint, runs it over tile rasters, and samples the
prediction at the pixel under each reference point.

Important behavior:

- `TestDataset` loads the full tile image and computes `(row, col)` from the
  point geometry using the raster transform.
- The model predicts a full segmentation map for the tile, but metrics are
  computed using only the pixel at `(row, col)` per point.

Outputs (in `--output_dir`):

- `assessment_results.csv` (per-point predictions)
- `assessment_metrics.json`
- `classification_report.csv`

### `assess.py`: product-level evaluation on a final raster

`assess.py` reads a completed land-cover raster product and compares sampled
values to reference points.

Outputs (in `--output_dir`):

- `assessment_results.csv` (per-point sampled values)
- `assessment_metrics.json`
- `classification_report.csv`
- `raster_class_distribution.csv` (distribution of values in the raster, excluding nodata)

## Sliding-Window (Tiled) Inference: Deep Dive

This section is a comprehensive, novice-friendly walkthrough of what happens when you run `python inference.py ...`.

Two files matter:

- **User-facing CLI**: `inference.py` (argument parsing, file discovery, multiprocessing wrapper)
- **Core implementation**: `src/mslandcover/inference.py` (raster loading, histogram matching, sliding-window tiling, blending, post-processing, GeoTIFF writing)

If you understand the details below, you’ll be able to:

- choose `--tile_size`, `--stride`, and `--batch_size` safely,
- avoid subtle band-order mistakes (the most common source of “it runs but looks wrong”),
- understand the difference between `.tif` and `.sid` inputs,
- run inference efficiently (single-GPU workstation, multi-core CPU, or HPC/SLURM).

### Quick mental model

Large rasters don’t fit into GPU memory as a single tensor. Sliding-window inference solves that by:

1. cutting the raster into many overlapping `tile_size × tile_size` chips,
2. running the segmentation model on chips in batches,
3. blending overlapping predictions back into one seamless output raster.

In this repo, blending is done using a **Gaussian weight mask** so tile edges contribute less than tile centers.

### End-to-end pipeline (what actually happens)

When you run `inference.py`, the process is:

1. **Find input rasters**
   - `inference.py` builds a list of paths using `glob(os.path.join(input_dir, file_pattern))`.
   - If you set `--index N`, it processes exactly one raster: `paths = [paths[N]]`.

2. **Per-raster processing (`process_single_raster`)**
   - Loads model weights (single model or ensemble).
   - Loads normalization tensors (`--mean_path`, `--std_path`).
   - Loads the raster and prepares it for model input (`load_raster_for_processing`).
   - Runs sliding-window inference (`RasterProcessor.process_raster`).
   - Converts probabilities → class IDs, applies nodata and optional post-processing.
   - Writes a **classified GeoTIFF** with an embedded class colormap.

### Input file types: GeoTIFF (`.tif`) vs MrSID (`.sid`)

The code supports both, but there are important differences.

#### `.tif` (GeoTIFF) inputs

This is the “first-class” path.

- Read using `rioxarray.open_rasterio(path)`.
- If `--match_histograms` is enabled, histogram matching is applied (details below).
- Band selection is applied via `in_data = in_data.sel(band=args.bands)`.

This path is what you should prefer unless you have a strong reason to keep `.sid`.

#### `.sid` (MrSID) inputs

`.sid` is handled by converting to a temporary GeoTIFF first:

- The code calls `mrsiddecode` to decode the `.sid` to a temporary `.tif`.
- If the decoded raster resolution is not exactly `(1.0, 1.0)`, it runs `gdalwarp` to resample to 1m using bilinear resampling.

Important caveats for `.sid`:

- **Histogram matching is disabled**: if you pass `--match_histograms` on a `.sid` input, the code raises `ValueError`.
- **Band selection is not applied in the current implementation** for `.sid` files. That means:
  - your `.sid` must already have exactly the channels your model expects, and
  - those channels must already be in the correct order.
- **Output filename uses the input filename**. The writer always outputs a GeoTIFF (`driver='GTiff'`), but the output path is `output_dir/<input_basename>`.
  - If your input is `something.sid`, you will currently write a GeoTIFF *named* `something.sid`.
  - Practically: if you want clean outputs, convert `.sid → .tif` ahead of time and run inference on `.tif`.

Also note: decoding `.sid` depends on external tooling (`mrsiddecode`, `gdalwarp`) and environment/library paths. If you’re on a new machine, you may need to install/configure those separately.

### Band ordering (IMPORTANT)

The segmentation models in this repo are configured for **3-channel CIR** inputs (Color Infrared):

- **Channel 0**: NIR
- **Channel 1**: Red
- **Channel 2**: Green

That is why the default CLI flag is:

```bash
--bands 4 1 2
```

For common NAIP 4-band imagery, the usual ordering is:

- band 1 = Red
- band 2 = Green
- band 3 = Blue
- band 4 = NIR

So `4 1 2` means “use NIR, then Red, then Green”, which matches the model’s expectation.

Why this matters:

- If you accidentally feed RGB (1,2,3) into a model trained on CIR (4,1,2), inference will **still run** but outputs will typically be nonsense.
- Band indices in this code are **1-indexed** because `rioxarray` uses rasterio-style band indexing.

Rule of thumb:

- If your model was trained on `--bands 4 1 2`, you must infer with the same `--bands`.
- If you change band order, you must re-train (or at least re-calibrate) because the input distribution changes.

### Histogram matching (`--match_histograms`)

Histogram matching is used to reduce distribution shift between datasets (e.g., 2016 vs 2023 imagery).

What it does here:

- Loads **precomputed per-band histograms** from package assets:
  - `src/mslandcover/assets/histograms/MS/2016.npy`
  - `src/mslandcover/assets/histograms/MS/2023.npy`
- Builds a per-band lookup table (LUT) using CDF matching.
- Applies the LUT to every pixel in each band.

Key nuances:

- It currently loads histograms with hard-coded arguments (`state='MS'`, years `2016 → 2023`). The CLI flag name is generic, but the implementation is specific to that use case.
- It is applied **before** `--bands` selection for `.tif` inputs. If your GeoTIFF has 4 bands, matching runs on all 4, then selection picks 3.
- The matching is tuned for **8-bit imagery**:
  - bins: `255`
  - value range: `(1, 255)`
  - output is cast back to `uint8`
  If your imagery is 16-bit or reflectance-scaled floats, histogram matching as implemented will not be appropriate without changes.

When to use it:

- Use `--match_histograms` when you trained on one “look” (e.g., 2023) but need to infer on a different “look” (e.g., 2016), and you have confirmed histograms exist for the source/target combination.

### Sliding-window inference mechanics

All sliding-window inference happens in `RasterProcessor`.

#### 1) Preprocessing: scaling and normalization

The model expects inputs that are:

- scaled to roughly $[0, 1]$ (default `--scale_factor 255.0`), then
- normalized per channel using `mean` and `std` loaded from `--mean_path` and `--std_path`:

```text
x_norm = (x - mean) / std
```

If `mean/std` don’t match the dataset you’re inferring on (or don’t match training), performance will degrade.

#### 2) Tile generation (`tile_size` and `stride`)

Tiles are generated by scanning a window across the raster:

- tile size: `tile_size × tile_size`
- step between windows: `stride`

Pseudo-logic (simplified):

```text
for y in range(0, height, stride):
  for x in range(0, width, stride):
    tile = raster[:, y:y+tile_size, x:x+tile_size]
```

Tiles are skipped if:

- they are entirely nodata, or
- they are not exactly full sized (edge tiles smaller than `tile_size`).

This leads to an important practical requirement:

- Your rasters should ideally have dimensions that allow full tiles across the full extent.
- If a raster’s width/height is not compatible with your `tile_size/stride`, some **valid edge pixels** may receive no predictions.

In many NAIP tiling pipelines, rasters are already cut to “nice” dimensions (often multiples of 256), so this may be fine. If you see suspicious class artifacts along the right/bottom edges, dimension/tile compatibility is one of the first things to check.

#### 3) Batching (`batch_size`)

Tiles are stacked into batches of size `--batch_size` and sent to the GPU.

- Larger `batch_size` is faster but uses more VRAM.
- If you hit CUDA OOM, reduce `--batch_size` first.

#### 4) Test-time augmentation (`--tta`) (current behavior)

If TTA is enabled, each tile is randomly:

- horizontally flipped with 50% probability,
- vertically flipped with 50% probability,
- rotated by $0, 90, 180, 270$ degrees.

The model predicts on the transformed tile, and then the transform is inverted on the output probabilities so everything aligns in the original orientation.

Two important nuances:

- This is **stochastic** (random per tile), so outputs are not guaranteed to be bitwise reproducible across runs.
- This is not the classic “multi-aug average” TTA; each tile gets one random augmentation.

Also, the current CLI argument is defined as `action='store_true', default=True`, which effectively means TTA is always on unless you edit the script. (Same issue exists for `--skip_existing`.)

#### 5) Gaussian blending (`gaussian_sigma`)

To avoid seam lines between tiles, predictions are blended with a Gaussian weight mask:

- The mask is highest in the center and lower near edges.
- Per-tile probabilities are multiplied by this mask before accumulation.

The `--gaussian_sigma` parameter controls how “flat” the weights are:

- **Smaller sigma** → stronger down-weighting of edges → smoother seams, but fewer pixels near tile borders strongly influence the final output.
- **Larger sigma** → flatter weights → uses more of the tile (including edges), which can be slightly sharper but may show more seam artifacts.

Implementation detail (useful for intuition):

- The code accumulates *weighted* class probabilities into an output tensor.
- It then normalizes across classes at each pixel. Because the model outputs a softmax (sums to 1 per pixel), this normalization effectively divides by “sum of weights contributed by overlapping tiles”, yielding a proper weighted average.

#### 6) From probabilities → class IDs

After blending, the code computes:

- `argmax` over the class probability dimension
- adds `+1` so class IDs are **1-indexed**

Then it applies a nodata mask:

- pixels where *all bands* equal the raster nodata value are set to class `0`.

Finally, optional post-processing:

- `--mode_filter_size > 0` applies a mode filter (majority vote in a sliding window) to reduce salt-and-pepper noise.

### Output raster format and semantics

Outputs are written with `rioxarray` as a single-band GeoTIFF:

- dtype: `uint8`
- nodata: `0`
- class IDs: `1..num_classes` (default: 1..8)
- compression: controlled by `--compress` (default: LZW)
- tiling/block size: controlled by `--block_size`

Then the file is re-opened with rasterio and a color table is attached using `LEGEND_COLORS_RGBA` from `src/mslandcover/config.py`.

### Running modes: single machine vs multiprocessing vs HPC

#### Single raster (debug / smoke test)

Use `--index` so you only process one raster and get a per-tile progress bar:

```bash
python inference.py \
  --input_dir /path/to/rasters \
  --output_dir /path/to/preds \
  --file_pattern "*.tif" \
  --index 0 \
  --num_processes 1
```

#### Processing a directory (one machine)

If you omit `--index`, the script processes all rasters matching `--file_pattern`.

- By default it picks up to 8 processes: `min(cpu_count - 1, len(paths), 8)`.
- Each process loads the model and runs inference.

Practical guidance:

- **If you are using a single GPU**, set `--num_processes 1`. Multiple processes will contend for the same GPU and often make things slower or cause OOM.
- Multiprocessing is more useful when running CPU-only inference (or when each process is pinned to a different GPU, which this script does not currently orchestrate automatically).

Example:

```bash
python inference.py \
  --input_dir /path/to/rasters \
  --output_dir /path/to/preds \
  --num_processes 1 \
  --tile_size 256 \
  --stride 128 \
  --batch_size 64
```

#### HPC / SLURM (recommended pattern)

The repository includes `inference.sh`, which demonstrates a common HPC strategy:

- submit many SLURM jobs,
- each job runs `python inference.py --index N --num_processes 1`,
- each job uses one GPU.

This pattern avoids multi-process GPU contention and scales cleanly.

One important indexing nuance:

- `--index` refers to the position in Python’s `glob(input_dir/file_pattern)` list.
- If you compute `INPUT_TOTAL` with `find` (as in many scripts), make sure the count and ordering match what `glob("*.tif")` will return.
  - Easiest: ensure the directory contains only the files you want (e.g., only `*.tif`).
  - Or: match the Python glob pattern when counting/iterating.

### Parameter tuning cheat-sheet

These are the most important knobs to tune sliding-window inference:

- `--tile_size`
  - Controls chip size.
  - Larger tiles reduce overhead but increase memory and can increase edge effects.
- `--stride`
  - Controls overlap.
  - Common choice: `stride = tile_size / 2` (50% overlap).
  - Smaller stride → more overlap → slower but often smoother.
- `--gaussian_sigma`
  - Controls blending smoothness.
  - For `tile_size=256`, values like `128–256` are typical starting points.
- `--batch_size`
  - Controls throughput/VRAM.
  - Reduce if you see CUDA OOM.
- `--mode_filter_size`
  - Use `0` to disable.
  - Try small odd values like `3` or `5` if you see speckle.
- `--bands`
  - Must match training. Default `4 1 2` is NAIP CIR (NIR, Red, Green).
- `--match_histograms`
  - Useful for cross-year domain shift (only implemented for `.tif` and MS 2016→2023 as currently written).

### Current implementation notes (things that can surprise you)

These are not “theory” notes — they reflect the current code exactly:

- The CLI exposes `--model`, but `process_single_raster` currently always constructs a `UNet` with a `ResNetBackboneUNet`. The `--model` choice is not wired into inference yet.
- `--weights_path` supports an ensemble *only* if it is already a Python list (e.g., the script’s default `glob(...)`). Passing multiple weights via CLI is not currently implemented.
- `--tta` and `--skip_existing` are defined as `store_true` with `default=True`, which makes them effectively always enabled unless the script is edited.
- Edge tiles that do not fit `tile_size × tile_size` are skipped; if your rasters are not tile-compatible, you may get unpredicted edge regions.

If you want, we can cleanly address these limitations (wire `--model`, fix `.sid` band selection/output suffix, make `--tta` truly toggleable, add optional padding for edge coverage), but that would be a code change beyond this README deep dive.

## What Artifacts To Expect

Pretraining (`pretrain.py`):

- Logs in `--log_dir/.../`:
  - `log.txt`, `history.csv`, `profiler.csv`, `model_complexity.json`
  - `checkpoint.pth` (resume)
  - `best_epoch.txt`, `finished.txt`
- Weights in `--weights_dir/.../`:
  - `<scheme>.pth` (best wrapper)
  - `<scheme>_last.pth` (exported encoder for fine-tuning)
- Shared stats:
  - `./weights/pretrain_mean.pth`, `./weights/pretrain_std.pth`

Fine-tuning (`finetune.py`):

- Logs in `--log_dir/`:
  - `log.txt`, `history.csv`, `profiler.csv`, `checkpoint.pth`, `finished.txt`
  - If `--test_dir` is set: `confusion_matrix.csv`, `classification_report.csv`, `assessment_metrics.json`
- Weights in `--output_dir/`:
  - `best_model.pth`

Inference (`inference.py`):

- Output rasters in `--output_dir/`:
  - single-band classified GeoTIFFs with class IDs and colormap

Evaluation:

- `test.py` writes `assessment_results.csv`, `assessment_metrics.json`, `classification_report.csv`
- `assess.py` writes `assessment_results.csv`, `assessment_metrics.json`, `classification_report.csv`, `raster_class_distribution.csv`

## If You Only Remember Three Files

1. `pretrain.py` for encoder learning.
2. `finetune.py` for supervised segmentation.
3. `src/mslandcover/models.py` for architecture internals and weight-key mapping context.

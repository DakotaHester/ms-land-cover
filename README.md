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
````

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

## What Artifacts To Expect

From pretraining runs:
- run log file (`log.txt`),
- history (`history.csv`),
- profiler (`profiler.csv`),
- model complexity report (`model_complexity.json`),
- rolling checkpoint (`checkpoint.pth`),
- best objective wrapper state (`<scheme>.pth`),
- exported encoder state (`<scheme>_last.pth`),
- completion marker (`finished.txt`).
`
From fine-tuning runs:
- best model checkpoint (`best_model.pth`),
- training history and profiler logs,
- optional test outputs (`confusion_matrix.csv`, `classification_report.csv`, `assessment_metrics.json`).

From inference runs:
- tiled/compressed classified GeoTIFF outputs with class colormap.

From assessment runs:
- sampled predictions and labels,
- confusion-matrix based summaries,
- overall/macro/weighted metric reports.

## If You Only Remember Three Files

1. `pretrain.py` for encoder learning.
2. `finetune.py` for supervised segmentation.
3. `src/mslandcover/models.py` for architecture internals and weight-key mapping context.

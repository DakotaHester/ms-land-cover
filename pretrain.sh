#!/bin/bash

# ======= Adjustable Constants =======
NUM_EPOCHS=100
INIT_LR=0.00001 # 1e-5
EARLY_STOPPING_PATIENCE=-1
WARMUP_EPOCHS=10 # 0 epochs of warmup
# REDUCE_LR_PATIENCE=0 # tolerate 4 bad epochs, reduce lr after 5 epochs of no improvement
TEMPERATURE=0.1 
NUM_WORKERS=48
SEED=1701
MINI_BATCH_SIZE=128
FULL_BATCH_SIZE=4096
# ====================================

# PRETRAIN_SCHEMES=("hires_simclr" "simclr")
PRETRAIN_SCHEMES=("byol")
# IMAGE_SIZES=(256 192 128)
# BATCH_SIZES=(128 256 512)
BANDS=(3)
# BATCH_SIZES=(128)
RAND_INIT_OPTIONS=(false)

LOAD_CHECKPOINT=false

COUNT=0

for rand_init in "${RAND_INIT_OPTIONS[@]}"; do
  # for batch in "${BATCH_SIZES[@]}"; do

  # if [[ "$batch" -eq 128 ]]; then
  #   size=256
  # elif [[ "$batch" -eq 256 ]]; then
  #   size=192
  # elif [[ "$batch" -eq 512 ]]; then
  #   size=128
  # fi

    for scheme in "${PRETRAIN_SCHEMES[@]}"; do

      for band in "${BANDS[@]}"; do
        # if [ "$scheme" == "byol" ] && [[ "$band" -eq 4 ]]; then
          # echo "Skipping: $scheme with 3 bands (not supported)"
          # continue
        # elif [ "$scheme" == "hires_byol" ] && [[ "$band" -eq 3 ]]; then
          # echo "Skipping: $scheme with 4 bands (not supported)"
          # continue
        # fi
        

        #   bands=$band
        if [[ "$scheme" == "hires_byol" ]]; then
          # bands=4
          size=256
        elif [[ "$scheme" == "byol" ]]; then
          # bands=3
          size=256
        fi

        LOG_DIR="./logs/pretrain/resnet101_202516/${scheme}_bands${band}_size${size}_randinit${rand_init}"
        WEIGHTS_DIR="./weights/resnet101_202516/${scheme}_bands${band}_size${size}_randinit${rand_init}"

        if [[ -f "$LOG_DIR/resnet101/$scheme/finished.txt" ]]; then
          echo "Skipping: $scheme bands=$band size=$size rand_init=$rand_init (already finished)"
#           continue
        elif [ "$LOAD_CHECKPOINT" == true ] && [[ -f "$LOG_DIR/resnet101/$scheme/checkpoint.pth" ]]; then
          echo "Loading checkpoint for: $scheme bands=$band size=$size rand_init=$rand_init"
          python pretrain.py \
            --pretrain_scheme "$scheme" \
            --n_bands "$band" \
            --image_size "$size" \
            --full_batch_size "$FULL_BATCH_SIZE" \
            --mini_batch_size "$MINI_BATCH_SIZE" \
            --log_dir "$LOG_DIR" \
            --weights_dir "$WEIGHTS_DIR" \
            --num_epochs "$NUM_EPOCHS" \
            --init_lr "$INIT_LR" \
            --warmup_epochs "$WARMUP_EPOCHS" \
            --num_workers "$NUM_WORKERS" \
            --seed "$SEED" \
            --load_checkpoint \
            $( [[ "$rand_init" == "true" ]] && echo "--rand_init" )
            COUNT=$((COUNT+1))
            continue
        fi

        mkdir -p "$LOG_DIR" "$WEIGHTS_DIR"
        echo "Run $COUNT | Running: $scheme bands=$band size=$size"
        # continue
          --pretrain_scheme "$scheme" \
            --n_bands "$band" \
            --image_size "$size" \
            --full_batch_size "$FULL_BATCH_SIZE" \
            --mini_batch_size "$MINI_BATCH_SIZE" \
            --log_dir "$LOG_DIR" \
            --weights_dir "$WEIGHTS_DIR" \
            --num_epochs "$NUM_EPOCHS" \
            --init_lr "$INIT_LR" \
            --warmup_epochs "$WARMUP_EPOCHS" \
            --num_workers "$NUM_WORKERS" \
            --seed "$SEED" \
            --load_checkpoint \
          --seed "$SEED" \
          $( [[ "$rand_init" == "true" ]] && echo "--rand_init" )
        COUNT=$((COUNT+1))
      # done
    done
  done
done

echo "Total test cases run: $COUNT"

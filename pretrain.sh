#!/bin/bash

# ======= Adjustable Constants =======
NUM_EPOCHS=100
INIT_LR=0.000001 # 1e-6
EARLY_STOPPING_PATIENCE=15
REDUCE_LR_PATIENCE=2 # tolerate 2 bad epochs, reduce lr after 3 epochs of no improvement
TEMPERATURE=0.5
NUM_WORKERS=32
SEED=1701
# ====================================

PRETRAIN_SCHEMES=("hires_simclr" "simclr")
# IMAGE_SIZES=(256 192 128)
# BATCH_SIZES=(128 256 512)
BANDS=(4 3)
BATCH_SIZES=(128 256)
RAND_INIT_OPTIONS=(false)

LOAD_CHECKPOINT=false

COUNT=0

for rand_init in "${RAND_INIT_OPTIONS[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do

  if [[ "$batch" -eq 128 ]]; then
    size=256
  elif [[ "$batch" -eq 256 ]]; then
    size=192
  elif [[ "$batch" -eq 512 ]]; then
    size=128
  fi

    for scheme in "${PRETRAIN_SCHEMES[@]}"; do

      for band in "${BANDS[@]}"; do
        if [ "$scheme" == "simclr" ] && [[ "$band" -eq 4 ]]; then
          # echo "Skipping: $scheme with 3 bands (not supported)"
          continue
        fi

        bands=$band

        LOG_DIR="./logs/pretrain/resnet152_202505/${scheme}_bands${bands}_size${size}_batch${batch}_randinit${rand_init}"
        WEIGHTS_DIR="./weights/resnet152_202505/${scheme}_bands${bands}_size${size}_batch${batch}_randinit${rand_init}"

        if [[ -f "$LOG_DIR/resnet152/$scheme/finished.txt" ]]; then
          echo "Skipping: $scheme bands=$bands size=$size batch=$batch rand_init=$rand_init (already finished)"
          continue
        elif [ "$LOAD_CHECKPOINT" == true ] && [[ -f "$LOG_DIR/resnet152/$scheme/checkpoint.pth" ]]; then
          echo "Loading checkpoint for: $scheme bands=$bands size=$size batch=$batch rand_init=$rand_init"
          python pretrain.py \
            --pretrain_scheme "$scheme" \
            --n_bands "$bands" \
            --image_size "$size" \
            --full_batch_size "$batch" \
            --mini_batch_size "$batch" \
            --log_dir "$LOG_DIR" \
            --weights_dir "./weights/resnet152_202505/${scheme}_bands${bands}_size${size}_batch${batch}_randinit${rand_init}" \
            --num_epochs "$NUM_EPOCHS" \
            --init_lr "$INIT_LR" \
            --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
            --reduce_lr_patience "$REDUCE_LR_PATIENCE" \
            --temperature "$TEMPERATURE" \
            --num_workers "$NUM_WORKERS" \
            --seed "$SEED" \
            --load_checkpoint \
            $( [[ "$rand_init" == "true" ]] && echo "--rand_init" )
            COUNT=$((COUNT+1))
            continue
        fi

        mkdir -p "$LOG_DIR" "$WEIGHTS_DIR"
        echo "Run $COUNT | Running: $scheme bands=$bands size=$size batch=$batch"
        # continue
        python pretrain.py \
          --pretrain_scheme "$scheme" \
          --n_bands "$bands" \
          --image_size "$size" \
          --full_batch_size "$batch" \
          --mini_batch_size "$batch" \
          --log_dir "$LOG_DIR" \
          --weights_dir "$WEIGHTS_DIR" \
          --num_epochs "$NUM_EPOCHS" \
          --init_lr "$INIT_LR" \
          --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
          --reduce_lr_patience "$REDUCE_LR_PATIENCE" \
          --temperature "$TEMPERATURE" \
          --num_workers "$NUM_WORKERS" \
          --seed "$SEED" \
          $( [[ "$rand_init" == "true" ]] && echo "--rand_init" )
        COUNT=$((COUNT+1))
      done
    done
  done
done

echo "Total test cases run: $COUNT"

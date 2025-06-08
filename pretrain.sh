#!/bin/bash

# ======= Adjustable Constants =======
NUM_EPOCHS=50
INIT_LR=0.0001 # 1e-4
EARLY_STOPPING_PATIENCE=5
REDUCE_LR_PATIENCE=0 # tolerate 0 bad epochs, reduce lr after 1 epochs of no improvement
TEMPERATURE=0.1 
NUM_WORKERS=48
SEED=1701
MINI_BATCH_SIZE=256
FULL_BATCH_SIZE=4096
# ====================================

# PRETRAIN_SCHEMES=("hires_simclr" "simclr")
PRETRAIN_SCHEMES=("byol" "hires_byol")
# IMAGE_SIZES=(256 192 128)
# BATCH_SIZES=(128 256 512)
BANDS=(4 3)
BATCH_SIZES=(128)
RAND_INIT_OPTIONS=(false)

LOAD_CHECKPOINT=false

COUNT=0

for rand_init in "${RAND_INIT_OPTIONS[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do

  # if [[ "$batch" -eq 128 ]]; then
  #   size=256
  # elif [[ "$batch" -eq 256 ]]; then
  #   size=192
  # elif [[ "$batch" -eq 512 ]]; then
  #   size=128
  # fi

    for scheme in "${PRETRAIN_SCHEMES[@]}"; do

      for band in "${BANDS[@]}"; do
        if [ "$scheme" == "byol" ] && [[ "$band" -eq 4 ]]; then
          # echo "Skipping: $scheme with 3 bands (not supported)"
          continue
        fi

      #   bands=$band
      if [[ "$scheme" == "hires_byol" ]]; then
        # bands=4
        size=192
      elif [[ "$scheme" == "byol" ]]; then
        # bands=3
        size=256
      fi

      LOG_DIR="./logs/pretrain/resnet152_202506/${scheme}_bands${bands}_size${size}_batch${batch}_randinit${rand_init}"
      WEIGHTS_DIR="./weights/resnet152_202506/${scheme}_bands${bands}_size${size}_batch${batch}_randinit${rand_init}"

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
          --weights_dir "./weights/resnet152_202506/${scheme}_bands${bands}_size${size}_batch${batch}_randinit${rand_init}" \
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
      # done
    done
  done
done

echo "Total test cases run: $COUNT"

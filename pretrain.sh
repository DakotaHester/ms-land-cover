#!/bin/bash

# This bash file submits a series of pretraining jobs in a loop.

# ======= Adjustable Constants =======
NUM_EPOCHS=300
INIT_LR=0.001 # 1e-3
WARMUP_EPOCHS=10 # 0 epochs of warmup
NUM_WORKERS=48
SEED=1701
MINI_BATCH_SIZE=256
FULL_BATCH_SIZE=4096
# ====================================

PRETRAIN_SCHEMES=("moco" "dino")
RAND_INIT_OPTIONS=(false true)

LOAD_CHECKPOINT=false

COUNT=0

for scheme in "${PRETRAIN_SCHEMES[@]}"; do
  for rand_init in "${RAND_INIT_OPTIONS[@]}"; do

        size=256

        LOG_DIR="./logs/pretrain/resnet101_20250720/${scheme}_bands3_size${size}_randinit${rand_init}"
        WEIGHTS_DIR="./weights/resnet101_20250720/${scheme}_bands3_size${size}_randinit${rand_init}"

        if [[ -f "$LOG_DIR/resnet101/$scheme/finished.txt" ]]; then
          echo "Skipping: $scheme bands=3 size=$size rand_init=$rand_init (already finished)"
          continue
          
        elif [ "$LOAD_CHECKPOINT" == true ] && [[ -f "$LOG_DIR/resnet101/$scheme/checkpoint.pth" ]]; then
          echo "Loading checkpoint for: $scheme bands=3 size=$size rand_init=$rand_init"
          python pretrain.py \
            --pretrain_scheme "$scheme" \
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
        echo "Run $COUNT | Running: $scheme bands=3 size=$size"
        # continue
        python pretrain.py \
            --pretrain_scheme "$scheme" \
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
            $( [[ "$rand_init" == "true" ]] && echo "--rand_init" )
        COUNT=$((COUNT+1))
  done
done

echo "Total test cases run: $COUNT"

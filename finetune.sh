#!/bin/bash

# =========================
# SLURM & ENVIRONMENT SETUP
# =========================
PARTITION="gpu-a100"
ACCOUNT="research-abe"
MEMORY="16G"
N_TASKS="4"
TIME="48:00:00"
GRES="gpu:a100_1g.10gb:1"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="finetune.py"
BASE_LOG_DIR="./logs/finetune_20250620"
BASE_WEIGHTS_DIR="./weights/finetune_20250620"
SLURM_SCRIPT_DIR="./slurm_scripts/finetune_20250620"
SPLIT_DIR="./data/splits"
MAX_JOBS=10

# =========================
# MODEL & TRAINING CONFIGURATION
# =========================
FREEZE_DECODER=false         # --freeze_decoder (not supported atm)
MINI_BATCH_SIZE=16           # --mini_batch_size
FULL_BATCH_SIZE=16           # --full_batch_size
LR=1e-5                      # --lr
NUM_EPOCHS=1000              # --num_epochs
EARLY_STOPPING_PATIENCE=50   # --early_stopping_patience
REDUCE_LR_PATIENCE=10        # --reduce_lr_patience
PRELOAD=false                # --load_data_from_disk
NUM_WORKERS=2                # --num_workers
SEED=1701                    # --seed
LOAD_CHECKPOINT=false        # --load_checkpoint

# Advanced/optional loss and sampling configs
MINIMUM_CLASS_PROPORTION=0.0         # --minimum_class_proportion
OVERSAMPLE_FACTOR=2                  # --oversample_factor
MINIMUM_OVERSAMPLE_RATIO_FACTOR=2.0  # --minimum_oversample_ratio_factor
ALPHA_POWER=0.0                      # --alpha_power
FOCAL_GAMMA=2.0                      # --focal_gamma

# Ensure script directory exists
mkdir -p "$SLURM_SCRIPT_DIR"

# =========================
# PRETRAIN SCHEMES TO TEST
# =========================
MODELS=("unet" "deeplabv3plus" "linear_probe")
# PRETRAIN_SCHEMES=("hires_simclr" "simclr" "imagenet")
# PRETRAIN_SCHEMES=("imagenet" "simclr")
PRETRAIN_SCHEMES=("none" "imagenet" "byol")
BANDS=(3)  # 4 bands for hires_simclr, 3 bands for simclr
PRETRAIN_SIZES=(256)
FREEZE_ENCODERS=(false true) # Freeze encoder options

# Dataset sizes and folds
declare -A FOLDS
FOLDS[250]=4
FOLDS[500]=6
FOLDS[750]=4

pre_size=256
pre_batch=128

# =========================
# JOB SUBMISSION LOOP
# =========================
COUNT=0

for model in "${MODELS[@]}"; do
    for n_train in 250 500 750; do
    n_folds=${FOLDS[$n_train]}

        for fold in $(seq 1 $n_folds); do
            for scheme in "${PRETRAIN_SCHEMES[@]}"; do
                for pre_size in "${PRETRAIN_SIZES[@]}"; do
                    
                    if [[ "$scheme" -eq "imagenet" ]]; then
                        pre_size=256  # Imagenet always uses 256 (imagenet technically uses 224, but we use 256 for simplicity here)
                    fi

                    if [[ "$pre_size" -eq 256 ]]; then
                        pre_batch=128
                    elif [[ "$pre_size" -eq 192 ]]; then
                        pre_batch=256
                    elif [[ "$pre_size" -eq 128 ]]; then
                        pre_batch=512
                    fi

                    for bands in "${BANDS[@]}"; do
                        for freeze_encoder in "${FREEZE_ENCODERS[@]}"; do

                            if [[ "$scheme" == "hires_simclr" && "$bands" -ne 4 ]]; then
                                continue
                            elif [[ "$scheme" == "simclr" && "$bands" -ne 3 ]]; then
                                continue
                            fi

                            # Set encoder weights path or keyword
                            if [[ "$scheme" == "imagenet" ]]; then
                                ENCODER_WEIGHTS="imagenet"
                            elif ENCODER_WEIGHTS="none"; then
                                ENCODER_WEIGHTS="none"
                            else
                                ENCODER_WEIGHTS="./weights/resnet101_20250616_BACKUP/${scheme}_bands${bands}_size${pre_size}_randinitfalse/resnet101/${scheme}_last.pth"
                            fi

                            # Unique log/output directories for this job
                            JOB_NAME="${model}_ft_${scheme}_bands${bands}_size${pre_size}_batch${pre_batch}_randinitfalse_frozenencoder${freeze_encoder}_n${n_train}_fold${fold}"
                            BASE_DIR="${model}/${scheme}/${bands}_bands/presize_${pre_size}/prebatch_${pre_batch}/randinit_false/frozenencoder_${freeze_encoder}/${n_train}/fold_${fold}"
                            JOB_LOG_DIR="${BASE_LOG_DIR}/${BASE_DIR}"
                            JOB_WEIGHTS_DIR="${BASE_WEIGHTS_DIR}/${BASE_DIR}"
                            mkdir -p "$JOB_LOG_DIR" "$JOB_WEIGHTS_DIR"

                            SLURM_SCRIPT="${JOB_LOG_DIR}/job.slurm"
                            LOG_FILE="${JOB_LOG_DIR}/slurm.out"
                            FINISHED_FILE="${JOB_LOG_DIR}/finished.txt"

                            if [[ -f "$FINISHED_FILE" ]]; then
                                echo "Skipping $JOB_NAME (already finished)"
                                continue
                            fi

                            # Wait if too many jobs are queued or running
                            while true; do
                                TOTAL_JOBS=$(squeue -u "$USER" | tail -n +2 | wc -l)
                                if [[ "$TOTAL_JOBS" -lt "$MAX_JOBS" ]]; then
                                    break
                                else
                                    echo "[$(date)] $TOTAL_JOBS jobs in queue. Waiting to submit $JOB_NAME..."
                                    sleep 60
                                fi
                            done

                            # Create SLURM script
                            cat > "$SLURM_SCRIPT" <<EOL
#!/bin/bash
#SBATCH -N 1
#SBATCH -n $N_TASKS
#SBATCH --mem=$MEMORY
#SBATCH -p $PARTITION
#SBATCH -A $ACCOUNT
#SBATCH -t $TIME
#SBATCH --gres=$GRES
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_FILE
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$MAIL_USER

ml cuda
ml python/3.10.8
source $PYTHON_ENV
export CUDA_VISIBLE_DEVICES=0

echo "========== SLURM JOB INFO =========="
echo "Job Name: $JOB_NAME"
echo "Pretrain scheme: $scheme"
echo "Bands: $bands"
echo "n_train: $n_train"
echo "fold: $fold"
echo "Encoder weights: $ENCODER_WEIGHTS"
echo "Log dir: $JOB_LOG_DIR"
echo "Weights dir: $JOB_WEIGHTS_DIR"
echo "===================================="
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: \$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"
echo "===================================="

python $SCRIPT_NAME \
--model $model \
--encoder_weights "$ENCODER_WEIGHTS" \
--split_dir "$SPLIT_DIR" \
--n_train_samples $n_train \
--fold $fold \
--n_bands $bands \
--mini_batch_size $MINI_BATCH_SIZE \
--full_batch_size $FULL_BATCH_SIZE \
--lr $LR \
--num_epochs $NUM_EPOCHS \
--early_stopping_patience $EARLY_STOPPING_PATIENCE \
--reduce_lr_patience $REDUCE_LR_PATIENCE \
--log_dir "$JOB_LOG_DIR" \
--output_dir "$JOB_WEIGHTS_DIR" \
--num_workers $NUM_WORKERS \
--seed $SEED \
$( [[ "$freeze_encoder" == true ]] && echo "--freeze_encoder" ) \
$( [[ "$FREEZE_DECODER" == true ]] && echo "--freeze_decoder" ) \
$( [[ "$PRELOAD" == true ]] && echo "--preload" ) \
$( [[ "$LOAD_CHECKPOINT" == true ]] && echo "--load_checkpoint" ) \
--minimum_class_proportion $MINIMUM_CLASS_PROPORTION \
--oversample_factor $OVERSAMPLE_FACTOR \
--minimum_oversample_ratio_factor $MINIMUM_OVERSAMPLE_RATIO_FACTOR \
--alpha_power $ALPHA_POWER \
--focal_gamma $FOCAL_GAMMA
EOL

                            # Submit the job
                            sbatch "$SLURM_SCRIPT"
                            echo "Submitted $JOB_NAME (log: $LOG_FILE)"
                            COUNT=$((COUNT+1))
                            sleep 5 # Slight delay to avoid race conditions

                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Total jobs submitted: $COUNT"

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
SCRIPT_NAME="test.py"
BASE_LOG_DIR="./logs/finetune_20250607"
BASE_WEIGHTS_DIR="./weights/finetune_20250607"
SLURM_SCRIPT_DIR="./slurm_scripts/finetune_20250607"
MAX_JOBS=10

BATCH_SIZE=16  # --batch_size

# Ensure script directory exists
mkdir -p "$SLURM_SCRIPT_DIR"

# =========================
# PRETRAIN SCHEMES TO TEST
# =========================
MODELS=("unet" "deeplabv3plus")
# PRETRAIN_SCHEMES=("hires_simclr" "simclr" "imagenet")
# PRETRAIN_SCHEMES=("imagenet" "simclr")
PRETRAIN_SCHEMES=("hires_byol" "imagenet" "byol")
BANDS=(3)  # 4 bands for hires_simclr, 3 bands for simclr
# PRETRAIN_SIZES=(256)
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
                            else
                                ENCODER_WEIGHTS="./weights/resnet101_202506/${scheme}_bands${bands}_size${pre_size}_randinitfalse/resnet101/${scheme}.pth"
                            fi

                            # Unique log/output directories for this job
                            JOB_NAME="${model}_test_${scheme}_bands${bands}_size${pre_size}_batch${pre_batch}_randinitfalse_frozenencoder${freeze_encoder}_n${n_train}_fold${fold}"
                            BASE_DIR="${model}/${scheme}/${bands}_bands/presize_${pre_size}/prebatch_${pre_batch}/randinit_false/frozenencoder_${freeze_encoder}/${n_train}/fold_${fold}"
                            JOB_LOG_DIR="${BASE_LOG_DIR}/${BASE_DIR}/test"
                            JOB_WEIGHTS_PATH="${BASE_WEIGHTS_DIR}/${BASE_DIR}/best_model.pth"
                            mkdir -p "$JOB_LOG_DIR"

                            SLURM_SCRIPT="${JOB_LOG_DIR}/job.slurm"
                            LOG_FILE="${JOB_LOG_DIR}/slurm.out"
                            FINISHED_FILE="${JOB_LOG_DIR}/classification_report.csv"

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

python $SCRIPT_NAME \
    --model "$model" \
    --model_weights "$JOB_WEIGHTS_PATH" \
    --n_bands "$bands" \
    --output_dir "$JOB_LOG_DIR" \
    --batch_size "$BATCH_SIZE" \
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

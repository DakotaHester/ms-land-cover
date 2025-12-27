#!/bin/bash

# =========================
# HOSTNAME CHECK & EXECUTION MODE
# =========================
CURRENT_HOSTNAME=$(hostname)
if [[ "$CURRENT_HOSTNAME" == "ptolemy-login-1.arc.msstate.edu" ]]; then
    EXECUTION_MODE="slurm"
    echo "Running on SLURM cluster: $CURRENT_HOSTNAME"
else
    EXECUTION_MODE="direct"
    echo "Running directly on: $CURRENT_HOSTNAME"
fi

# =========================
# SLURM & ENVIRONMENT SETUP
# =========================
PARTITION="gpu-a100-mig7"
ACCOUNT="research-abe"
MEMORY="16G"
N_TASKS="8"
TIME="24:00:00"
GRES="gpu:a100_1g.10gb:1"
#GRESs="gpu:nvidia_a100_80gb_pcie_1g.10gb:1"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="finetune.py"
BASE_LOG_DIR="./logs/finetune_20250720"
BASE_WEIGHTS_DIR="./weights/finetune_20250720"
SLURM_SCRIPT_DIR="./slurm_scripts/finetune_20250720"
SPLIT_DIR="./data/splits"
MAX_JOBS=10

# =========================
# MODEL & TRAINING CONFIGURATION
# =========================
FREEZE_DECODER=false         # --freeze_decoder (not supported atm)
MINI_BATCH_SIZE=32           # --mini_batch_size
FULL_BATCH_SIZE=32           # --full_batch_size
LR=1e-4                      # --lr
NUM_EPOCHS=1000              # --num_epochs
EARLY_STOPPING_PATIENCE=50   # --early_stopping_patience
REDUCE_LR_PATIENCE=15        # --reduce_lr_patience
PRELOAD=false                # --load_data_from_disk
NUM_WORKERS=$N_TASKS         # --num_workers
SEED=1701                    # --seed
LOAD_CHECKPOINT=false        # --load_checkpoint

# Advanced/optional loss and sampling configs
MINIMUM_CLASS_PROPORTION=0.0         # --minimum_class_proportion
OVERSAMPLE_FACTOR=2                  # --oversample_factor
MINIMUM_OVERSAMPLE_RATIO_FACTOR=2.0  # --minimum_oversample_ratio_factor
ALPHA_POWER=0.0                      # --alpha_power
FOCAL_GAMMA=2.0                      # --focal_gamma

# Ensure script directory exists (only for SLURM mode)
if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    mkdir -p "$SLURM_SCRIPT_DIR"
fi

# =========================
# PRETRAIN SCHEMES TO TEST
# =========================
# MODELS=("unet" "deeplabv3plus" "linear_probe")
MODELS=("deeplabv3plus" "unet" "attention_unet" "linear_probe" "upernet"  "pan" "fcn")
# MODELS=("unet")
# MODELS=("danet" "pan")
PRETRAIN_SCHEMES=("dino" "dino_randinit"))
# FREEZE_ENCODERS=(false true) # Freeze encoder options

# Dataset sizes and folds
declare -A FOLDS
FOLDS[250]=4
FOLDS[500]=6
FOLDS[750]=4

pre_batch=128

# =========================
# FUNCTION TO RUN PYTHON SCRIPT
# =========================
run_python_job() {
    local model=$1
    local scheme=$2
    local n_train=$3
    local fold=$4
    local freeze_encoder=$5
    local encoder_weights=$6
    local job_log_dir=${7}
    local job_weights_dir=${8}
    
    echo "========== RUNNING JOB DIRECTLY =========="
    echo "Model: $model"
    echo "Pretrain scheme: $scheme"
    echo "n_train: $n_train"
    echo "fold: $fold"
    echo "Encoder weights: $encoder_weights"
    echo "Log dir: $job_log_dir"
    echo "Weights dir: $job_weights_dir"
    echo "=========================================="
    
    # Activate Python environment if not on SLURM
    if [[ -f "$PYTHON_ENV" ]]; then
        source "$PYTHON_ENV"
    fi
    
    # Run the Python script directly
    python "$SCRIPT_NAME" \
        --model "$model" \
        --encoder_weights "$encoder_weights" \
        --split_dir "$SPLIT_DIR" \
        --n_train_samples "$n_train" \
        --fold "$fold" \
        --mini_batch_size "$MINI_BATCH_SIZE" \
        --full_batch_size "$FULL_BATCH_SIZE" \
        --lr "$LR" \
        --num_epochs "$NUM_EPOCHS" \
        --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
        --reduce_lr_patience "$REDUCE_LR_PATIENCE" \
        --log_dir "$job_log_dir" \
        --output_dir "$job_weights_dir" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED" \
        $( [[ "$freeze_encoder" == true ]] && echo "--freeze_encoder" ) \
        $( [[ "$FREEZE_DECODER" == true ]] && echo "--freeze_decoder" ) \
        $( [[ "$PRELOAD" == true ]] && echo "--preload" ) \
        $( [[ "$LOAD_CHECKPOINT" == true ]] && echo "--load_checkpoint" ) \
        --minimum_class_proportion "$MINIMUM_CLASS_PROPORTION" \
        --oversample_factor "$OVERSAMPLE_FACTOR" \
        --minimum_oversample_ratio_factor "$MINIMUM_OVERSAMPLE_RATIO_FACTOR" \
        --alpha_power "$ALPHA_POWER" \
        --focal_gamma "$FOCAL_GAMMA"
}

# =========================
# JOB SUBMISSION LOOP
# =========================
COUNT=0

n_folds=6
for fold in $(seq 1 $n_folds); do
    for n_train in 250 500 750; do
        # if fold > 4 and n_train != 500, skip
        if [[ $fold -gt 4 && $n_train -ne 500 ]]; then
            # echo "Skipping fold $fold for n_train $n_train (only 500 samples allowed
            continue
            # echo "Rerunning"
        fi
            for scheme in "${PRETRAIN_SCHEMES[@]}"; do
                for model in "${MODELS[@]}"; do
                # for freeze_encoder in "${FREEZE_ENCODERS[@]}"; do

                    # Set encoder weights path or keyword
                    if [[ "$scheme" == "byol" ]]; then
                        ENCODER_WEIGHTS="./weights/resnet101_20250624/byol_bands3_size256_randinitfalse/resnet101/byol_last.pth"
                    elif [[ "$scheme" == "byol_randinit" ]]; then
                        ENCODER_WEIGHTS="./weights/resnet101_20250624/byol_bands3_size256_randinittrue/resnet101_randinit/byol_last.pth"
                    elif [[ "$scheme" == "moco" ]]; then
                        ENCODER_WEIGHTS="./weights/resnet101_20250720/moco_bands3_size256_randinitfalse/resnet101/moco_last.pth"
                    elif [[ "$scheme" == "moco_randinit" ]]; then
                        ENCODER_WEIGHTS="./weights/resnet101_20250720/moco_bands3_size256_randinittrue/resnet101_randinit/moco_last.pth"
                    elif [[ "$scheme" == "dino" ]]; then
                        ENCODER_WEIGHTS="./weights/resnet101_20250720/dino_bands3_size256_randinitfalse/resnet101/dino_last.pth"
                    elif [[ "$scheme" == "dino_randinit" ]]; then
                        ENCODER_WEIGHTS="./weights/resnet101_20250720/dino_bands3_size256_randinittrue/resnet101_randinit/dino_last.pth"
                    else
                        ENCODER_WEIGHTS="$scheme"
                    fi
                    
                    # if linear_probe, set freeze_encoder to true
                    if [[ "$model" == "linear_probe" ]]; then
                        freeze_encoder=true
                    else
                        freeze_encoder=false
                    fi

                    # Unique log/output directories for this job
                    JOB_NAME="${model}_ft_${scheme}_randinitfalse_frozenencoder${freeze_encoder}_n${n_train}_fold${fold}"
                    BASE_DIR="${model}/${scheme}/randinit_false/frozenencoder_${freeze_encoder}/${n_train}/fold_${fold}"
                    JOB_LOG_DIR="${BASE_LOG_DIR}/${BASE_DIR}"
                    JOB_WEIGHTS_DIR="${BASE_WEIGHTS_DIR}/${BASE_DIR}"
                    mkdir -p "$JOB_LOG_DIR" "$JOB_WEIGHTS_DIR"

                    FINISHED_FILE="${JOB_LOG_DIR}/finished.txt"

                    # if [[ -f "$FINISHED_FILE" ]]; then
                    #     echo "Skipping $JOB_NAME (already finished)"
                    #     continue
                    # fi

                    if [[ "$EXECUTION_MODE" == "slurm" ]]; then
                        # SLURM EXECUTION MODE
                        SLURM_SCRIPT="${JOB_LOG_DIR}/job.slurm"
                        LOG_FILE="${JOB_LOG_DIR}/slurm.out"

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

python $SCRIPT_NAME \\
--model $model \\
--encoder_weights "$ENCODER_WEIGHTS" \\
--split_dir "$SPLIT_DIR" \\
--n_train_samples $n_train \\
--fold $fold \\
--mini_batch_size $MINI_BATCH_SIZE \\
--full_batch_size $FULL_BATCH_SIZE \\
--lr $LR \\
--num_epochs $NUM_EPOCHS \\
--early_stopping_patience $EARLY_STOPPING_PATIENCE \\
--reduce_lr_patience $REDUCE_LR_PATIENCE \\
--log_dir "$JOB_LOG_DIR" \\
--output_dir "$JOB_WEIGHTS_DIR" \\
--num_workers $NUM_WORKERS \\
--seed $SEED \\
\$( [[ "$freeze_encoder" == true ]] && echo "--freeze_encoder" ) \\
\$( [[ "$FREEZE_DECODER" == true ]] && echo "--freeze_decoder" ) \\
\$( [[ "$PRELOAD" == true ]] && echo "--preload" ) \\
\$( [[ "$LOAD_CHECKPOINT" == true ]] && echo "--load_checkpoint" ) \\
--minimum_class_proportion $MINIMUM_CLASS_PROPORTION \\
--oversample_factor $OVERSAMPLE_FACTOR \\
--minimum_oversample_ratio_factor $MINIMUM_OVERSAMPLE_RATIO_FACTOR \\
--alpha_power $ALPHA_POWER \\
--focal_gamma $FOCAL_GAMMA

python test.py \\
    --model "$model" \\
    --model_weights "$JOB_WEIGHTS_DIR/best_model.pth" \\
    --output_dir "$JOB_LOG_DIR/test" \\
    --batch_size $FULL_BATCH_SIZE
EOL

                        # Submit the job
                        sbatch "$SLURM_SCRIPT"
                        echo "Submitted $JOB_NAME (log: $LOG_FILE)"
                        COUNT=$((COUNT+1))
                        sleep 1 # Slight delay to avoid race conditions

                    else
                        # DIRECT EXECUTION MODE
                        run_python_job "$model" "$scheme" "$n_train" "$fold" "$freeze_encoder" "$ENCODER_WEIGHTS" "$JOB_LOG_DIR" "$JOB_WEIGHTS_DIR"
                        COUNT=$((COUNT+1))

                        # Run test.py immediately after finetune.py in direct mode
                        python test.py \
                            --model "$model" \
                            --model_weights "$JOB_WEIGHTS_DIR/best_model.pth" \
                            --output_dir "$JOB_LOG_DIR/test" \
                            --batch_size $FULL_BATCH_SIZE
                        
                        # Create finished file to mark completion
                        echo "$(date): Job completed successfully" > "$FINISHED_FILE"
                    fi

                # done
            done
        done
    done
done

if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    echo "Total SLURM jobs submitted: $COUNT"
else
    echo "Total jobs run directly: $COUNT"
fi

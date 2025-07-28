#/bin/bash

PARTITION="gpu-a100-mig7"
ACCOUNT="research-abe"
MEMORY="16G"
N_TASKS="1"
TIME="24:00:00"
GRES="gpu:a100_1g.10gb:1"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="inference.py"
BASE_LOG_DIR="./logs/inference_20250720"
SLURM_SCRIPT_DIR="./slurm_scripts/inference_20250720"
INPUT_BASE_DIR="/scratch/ptolemy/users/dh2306/mslc/imagery"
OUTPUT_BASE_DIR="/scratch/ptolemy/users/dh2306/mslc/out_20250720"
MAX_JOBS=10

# WEIGHTS_PATH="./weights/finetune_20250707/unet/byol/randinit_false/frozenencoder_false/500/fold_2/best_model.pth"
# new weights defined in model

states=("ms")
# years=("2016" "2023")
years=("2023")

for state in "${states[@]}"; do
    for year in "${years[@]}"; do
        INPUT_DIR="${INPUT_BASE_DIR}/${state}/${year}"
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${state}/${year}"
        LOG_DIR="${BASE_LOG_DIR}/${state}/${year}"

        # Create necessary directories
        mkdir -p "${LOG_DIR}"
        mkdir -p "${OUTPUT_DIR}"

        # if state == "ms" and year == "2016', pass --match_histograms
        if [[ "$state" == "ms" && "$year" == "2016" ]]; then
            MATCH_HISTOGRAMS="--match_histograms"
        else
            MATCH_HISTOGRAMS=""
        fi

        INPUT_TOTAL=$(find "${INPUT_DIR}" -type f | wc -l)
        echo "Total input directories for ${state} ${year}: ${INPUT_TOTAL}"
        for ((i=0; i<INPUT_TOTAL; i+=1)); do
            echo "${STATE} ${YEAR} - Processing tile ${i}"

            JOB_NAME="inference_${state}_${year}_tile_${i}"
            LOG_FILE="${LOG_DIR}/${i}.out"
            SLURM_SCRIPT="${SLURM_SCRIPT_DIR}/${JOB_NAME}.slurm"
            mkdir -p "${SLURM_SCRIPT_DIR}"

            # Wait if too many jobs are queued or running
            while true; do
                TOTAL_JOBS=$(squeue -u "$USER" | tail -n +2 | wc -l)
                if [[ "$TOTAL_JOBS" -lt "$MAX_JOBS" ]]; then
                    break
                else
                    # echo "[$(date)] $TOTAL_JOBS jobs in queue. Waiting to submit $JOB_NAME..."
                    sleep 5
                fi
            done

            cat > "${SLURM_SCRIPT}" <<EOL
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

python "${SCRIPT_NAME}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --weights_path "${WEIGHTS_PATH}" \
    --num_processes 1 \
    --index $i ${MATCH_HISTOGRAMS}

EOL

            sbatch "${SLURM_SCRIPT}"
            echo "Submitted job $JOB_NAME for tile ${i} in ${state} ${year}. Log: ${LOG_FILE}"
            sleep 1

        done
    done
done    

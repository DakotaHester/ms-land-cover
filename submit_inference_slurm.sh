#!/bin/bash

# Set common PBS parameters
PARTITION="gpu-a100"
ACCOUNT="research-abe"
NCPUS="64"
MEMORY="128G"
TIME="48:00:00"
GRES="gpu:a100_1g.10gb:1"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="inference.py"
LOG_DIR="./logs/slurm"

# SCRIPT_NAME="inference.py"
# PBS_LOG_DIR="./logs/inference"

START=74
END=75
for ((i=START; i<END; i++))
do
    JOB_NAME="inference_$i"
    SLURM_SCRIPT="./slurm_scripts/$JOB_NAME.pbs"
    cat > "$SLURM_SCRIPT" <<EOL
#!/bin/bash
#SBATCH -N 1
#SBATCH -n $NCPUS
#SBATCH --mem=$MEMORY
#SBATCH -p $PARTITION
#SBATCH -A $ACCOUNT
#SBATCH -t $TIME
#SBATCH --gres=$GRES
#SBATCH --job-name "$JOB_NAME"
#SBATCH --output=$LOG_DIR/${JOB_NAME}.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$MAIL_USER

ml cuda
ml python/3.10.8
source $PYTHON_ENV
export MSLC_INFERENCE_COUNTY_INDEX=$i
export CUDA_VISIBLE_DEVICES=0
python $SCRIPT_NAME
EOL
    # Submit the job
    sbatch "$SLURM_SCRIPT"
done
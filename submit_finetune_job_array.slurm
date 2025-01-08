#!/bin/bash

PARTITION="gpu-a100"
ACCOUNT="research-abe"
MEMORY="8G"
NCPUS="4"
TIME="48:00:00"
GRES="gpu:a100_1g.10gb:1"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="finetune.py"
LOG_DIR="./logs/slurm/finetune"
ARRAY="0-164%8"

#SBATCH -N 1
#SBATCH -n $NCPUS
#SBATCH --mem=$MEM
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
export CUDA_VISIBLE_DEVICES=0
python $SCRIPT_NAME
EOL
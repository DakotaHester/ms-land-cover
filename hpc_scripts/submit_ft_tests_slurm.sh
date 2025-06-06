#!/bin/bash

# Set common SLURM parameters
PARTITION="gpu-a100"
ACCOUNT="research-abe"
MEMORY="16G"
N_TASKS="2"
TIME="48:00:00"
GRES="gpu:a100_1g.10gb:1"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="finetune.py"
LOG_DIR="./logs/cpb_tests/slurm"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Loop through all model and pretrain_scheme combinations
for weights in randinit imagenet dae_hsv simclr dae_hsv_simclr
do
    for train_full_encoder in "" "--train_full_encoder"
    do
        if [ "$train_full_encoder" = "--train_full_encoder" ]; then
            JOB_NAME="dh2306_${weights}_ft_simple_decoder_test"
        else
            JOB_NAME="dh2306_${weights}_ft_test"
        fi
                                
        # Create a SLURM script for the job
        SLURM_SCRIPT="./slurm_scripts/${JOB_NAME}.slurm"
        mkdir -p "$(dirname "$SLURM_SCRIPT")"
                
        cat > "$SLURM_SCRIPT" <<EOL
#!/bin/bash
#SBATCH -N 1
#SBATCH -n $N_TASKS
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
export CUDA_VISIBLE_DEVICES=0
python $SCRIPT_NAME --weights $weights --n_layers_unfrozen 0 $train_full_encoder --num_workers $N_TASKS
EOL

		# Submit the job
	    sbatch "$SLURM_SCRIPT"
	done
done
	    

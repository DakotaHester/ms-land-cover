#!/bin/bash

# Set common SLURM parameters
PARTITION="gpu-a100"
ACCOUNT="research-abe"
MEMORY="10G"
TIME="48:00:00"
GRES="gpu:a100_1g.10gb:1"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="pretrain.py"
HDF5_PATH="./data/splits/pretrain.hdf5"
LOG_DIR="./logs/slurm"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Loop through all model and pretrain_scheme combinations
for model in hrnet_w18 hrnet_w48
do
	for pretrain_scheme in hsv dae dae_hsv
	do
		# Create a unique job name
		JOB_NAME="dh2306_${model}_${pretrain_scheme}_pretrain"
						        
	        # Create a SLURM script for the job
	        SLURM_SCRIPT="./slurm_scripts/${JOB_NAME}.slurm"
	        mkdir -p "$(dirname "$SLURM_SCRIPT")"
		       
	        cat > "$SLURM_SCRIPT" <<EOL
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
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
python $SCRIPT_NAME --model $model --pretrain_scheme $pretrain_scheme --pretrain_hdf5_path $HDF5_PATH
EOL

		# Submit the job
	        sbatch "$SLURM_SCRIPT"
	done
done
	    

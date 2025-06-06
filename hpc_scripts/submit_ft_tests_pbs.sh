#!/bin/bash

# Set common PBS parameters
QUEUE="biggpu"
NCPUS="2"
NGPUS="1"
MEMORY="16GB"
TIME="240:00:00"
MAIL_USER="dh2306@msstate.edu"
PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="finetune.py"
LOG_DIR="./logs/cpb_tests/pbs"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Loop through all model and pretrain_scheme combinations
for num_train_samples in 100 500 1000 5000
do
    for num_layers_unfrozen in 0 1 14
    do
        for weights in imagenet dae_hsv simclr dae_hsv_simclr
            JOB_NAME="dh2306_${weights}_ft_test_${num_train_samples}_${num_layers_unfrozen}"
                                    
            # Create a PBS script for the job
            PBS_SCRIPT="./pbs_scripts/${JOB_NAME}.pbs"
            mkdir -p "$(dirname "$PBS_SCRIPT")"
                    
            cat > "$PBS_SCRIPT" <<EOL
#!/bin/bash
#PBS -N $JOB_NAME
#PBS -q $QUEUE
#PBS -j oe
#PBS -o $LOG_DIR/$JOB_NAME.out
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEMORY
#PBS -l walltime=$TIME
#PBS -m abe
#PBS -M $MAIL_USER
 
cd \${PBS_O_WORKDIR}
module load cuda
module load python/3.10.8
source $PYTHON_ENV
export CUDA_VISIBLE_DEVICES=0
python $SCRIPT_NAME --weights $weights --n_layers_unfrozen $num_layers_unfrozen --num_train_samples $num_train_samples --num_workers $NCPUS
EOL

        # Submit the job
        qsub "$PBS_SCRIPT"
    done
done
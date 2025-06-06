#!/bin/bash

# Set common PBS parameters
QUEUE="biggpu"
NCPUS="64"
NGPUS="1"
MEMORY="256GB"
TIME="240:00:00"
MAIL_USER="dh2306@msstate.edu"

SCRIPT_NAME="inference.py"
PBS_LOG_DIR="./logs/inference"

START=74
END=75
for ((i=START; i<END; i++))
do
    JOB_NAME="inference_$i"
    PBS_SCRIPT="./pbs_scripts/$JOB_NAME.pbs"
    cat > "$PBS_SCRIPT" <<EOL
#!/bin/bash
#PBS -N $JOB_NAME
#PBS -q $QUEUE
#PBS -j oe
#PBS -o $PBS_LOG_DIR/$JOB_NAME.out
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEMORY
#PBS -l walltime=$TIME
#PBS -m abe
#PBS -M $MAIL_USER

cd \${PBS_O_WORKDIR}
export MSLC_INFERENCE_COUNTY_INDEX=$i
module load cuda10.2/toolkit
module load python
conda init
source ~/.bashrc
conda activate mslc
export CUDA_VISIBLE_DEVICES=0
python finetune.py
EOL
    # Submit the job
    qsub "$PBS_SCRIPT"
done
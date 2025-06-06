#!/bin/bash
#
## PBS Params
NAME="r2665-hrnet-pretrain"
QUEUE="biggpu"
NCPUS="4"
NGPUS="1"
MEMORY="128GB"
TIME="240:00:00"
MAIL_USER="dh2306@msstate.edu"
MINI_BATCH_SIZE=8
IMAGE_SIZE=96

PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="pretrain.py"
HDF5_PATH="/scratch/r2665/mslc/pretrain.hdf5"
LEARNING_RATE_FACTOR=1
FULL_BATCH_SIZE=1024
LOG_DIR="./logs/pbs"

mkdir -p "$LOG_DIR"

for model in resnet152
do
	
	for pretrain_scheme in dae_hsv simcle dae_hsv_simclr
	do
		JOB_NAME="${NAME}_${model}_${pretrain_scheme}"

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
module load cuda10.2/toolkit
module load python
conda init
source ~/.bashrc
conda activate mslc
python $SCRIPT_NAME --model $model --pretrain_scheme $pretrain_scheme --pretrain_hdf5_path $HDF5_PATH --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --num_workers $NCPUS --learning_rate_factor $LEARNING_RATE_FACTOR --image_size $IMAGE_SIZE --num_epochs 10

EOL
		qsub "$PBS_SCRIPT"
	done
done

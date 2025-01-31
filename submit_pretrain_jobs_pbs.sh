#!/bin/bash
#
## PBS Params
NAME="r2665-hrnet-pretrain"
QUEUE="biggpu"
NCPUS="8"
NGPUS="1"
MEMORY="128GB"
TIME="240:00:00"
MAIL_USER="dh2306@msstate.edu"

PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="pretrain.py"
HDF5_PATH="/scratch/r2665/mslc/pretrain.hdf5"
LEARNING_RATE_FACTOR=1
MINI_BATCH_SIZE=32
FULL_BATCH_SIZE=256
LOG_DIR="./logs/pbs"

mkdir -p "$LOG_DIR"

for model in unet
do
	for frozen_encoder in 0 # skip 1 as already submitted
	do
		for randinit in 1 # skip 0 as already submitted
		do

			if [ "$randinit" = "1" ]; then
				if [ "$frozen_encoder" = "1" ]; then
					continue
				fi
				RANDINIT_FLAG="--randinit"
				JOB_NAME="${NAME}_${model}_randinit"
			else
				RANDINIT_FLAG=""
				JOB_NAME="${NAME}_${model}"
			fi

			for pretrain_scheme in dae lab dae_lab
			do
				if [ "$frozen_encoder" = "1" ]; then
					FROZEN_ENCODER_FLAG="--frozen_encoder"
					JOB_NAME="${NAME}_${model}_${pretrain_scheme}_frozen"
				else
					FROZEN_ENCODER_FLAG=""
					JOB_NAME="${NAME}_${model}_${pretrain_scheme}"
				fi
				

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
python $SCRIPT_NAME --model $model --pretrain_scheme $pretrain_scheme --pretrain_hdf5_path $HDF5_PATH --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --num_workers $NCPUS --learning_rate_factor $LEARNING_RATE_FACTOR $FROZEN_ENCODER_FLAG $RANDINIT_FLAG
EOL
		    	qsub "$PBS_SCRIPT"
			done
	    done
    done
done

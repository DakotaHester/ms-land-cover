#!/bin/bash
#
## PBS Params
NAME="r2665-simclr-pretrain"
QUEUE="biggpu"
NCPUS="2"
NGPUS="1"
MEMORY="16GB"
TIME="240:00:00"
MAIL_USER="dh2306@msstate.edu"

PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="pretrain.py"
HDF5_PATH="/scratch/r2665/mslc/pretrain.hdf5"
# LEARNING_RATE_FACTOR=1
# MINI_BATCH_SIZE=32
# FULL_BATCH_SIZE=256
LOG_DIR="./logs/pbs"

mkdir -p "$LOG_DIR"

# stage 1 - encoder pretrain
model="resnet152d"
pretrain_scheme="simclr"
randinit_args=("" "--rand_init")
for randinit_arg in "${randinit_args[@]}"
do
    MINI_BATCH_SIZE=256
    FULL_BATCH_SIZE=256
    IMAGE_SIZE=128
    LEARNING_RATE=0.001

    WEIGHTS_DIR='./weights/simclr2_tests'
    JOB_NAME="${NAME}_${model}_${pretrain_scheme}"
    PROG_LOG_DIR='./logs/simclr2_tests'
    mkdir -p "$PROG_LOG_DIR"
    mkdir -p "$PROG_LOG_DIR/$model"

    python $SCRIPT_NAME --pretrain_scheme $pretrain_scheme --model $model --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --weights_dir $WEIGHTS_DIR --log_dir $PROG_LOG_DIR --image_size $IMAGE_SIZE --init_lr $LEARNING_RATE $randinit_arg

done

# stage 2 - decoder pretrain
model="att_unet"
encoder_weights=("simclr" "simclr_randinit" "imagenet" "randinit")
encoder_freeze_flags=("" "--frozen_encoder")
decoder_strategies=("dae" "dae_si")
for encoder_weights in "${encoder_weights[@]}"
do
    if [ "$encoder_weights" = "randinit" ]; then
        weights_argument="--rand_init"
    elif [ "$encoder_weights" = "simclr" ]; then
        weights_argument="--encoder_weights ./weights/simclr2_tests/resnet152d/simclr.pth"
    elif [ "$encoder_weights" = "simclr_randinit" ]; then
        weights_argument="--encoder_weights ./weights/simclr2_tests/resnet152d_randinit/simclr.pth"
    elif [ "$encoder_weights" = "imagenet" ]; then
        weights_argument=""
    fi

    for encoder_freeze_flags in "${encoder_freeze_flags[@]}"
    do

        # skip if encoder is frozen and randinit
        if [ "$encoder_weights" = "randinit" -a "$encoder_freeze_flags" = "--frozen_encoder" ]; then
            continue
        fi

        for decoder_strategy in "${decoder_strategies[@]}"
        do

            MINI_BATCH_SIZE=8
            FULL_BATCH_SIZE=8
            IMAGE_SIZE=256

            WEIGHTS_DIR='./weights/simclr2_tests'
            PROG_LOG_DIR='./logs/simclr2_tests_decoder'
            mkdir -p "$PROG_LOG_DIR"
            mkdir -p "$PROG_LOG_DIR/att_unet"

            python $SCRIPT_NAME --pretrain_scheme $decoder_strategy --model att_unet --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --weights_dir $WEIGHTS_DIR --log_dir $PROG_LOG_DIR --image_size $IMAGE_SIZE $weights_argument $encoder_freeze_flags

        done

    done

done

# for model in unet
# do
# 	for randinit in 0 1 # skip 0 as already submitted
# 	do
# 		for frozen_encoder in 0 1 # skip 1 as already submitted
# 		do
# 			for pretrain_scheme in dae lab dae_lab
# 			do

# 				JOB_NAME="${NAME}_${model}_${pretrain_scheme}"

# 				if [ "$randinit" = "1" ]; then
# 					if [ "$frozen_encoder" = "1" ]; then
# 						continue
# 					fi
# 					RANDINIT_FLAG="--rand_init"
# 					JOB_NAME="${JOB_NAME}_randinit"
# 				else
# 					RANDINIT_FLAG=""
# 				fi

# 				if [ "$frozen_encoder" = "1" ]; then
# 					FROZEN_ENCODER_FLAG="--frozen_encoder"
# 					JOB_NAME="${JOB_NAME}_frozen"
# 				else
# 					FROZEN_ENCODER_FLAG=""
# 				fi

# 				PBS_SCRIPT="./pbs_scripts/${JOB_NAME}.pbs"
# 				mkdir -p "$(dirname "$PBS_SCRIPT")"

# 				cat > "$PBS_SCRIPT" <<EOL
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
 
# cd \${PBS_O_WORKDIR}
# module load cuda10.2/toolkit
# module load python
# conda init
# source ~/.bashrc
# conda activate mslc
# python $SCRIPT_NAME --model $model --pretrain_scheme $pretrain_scheme --pretrain_hdf5_path $HDF5_PATH --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --num_workers $NCPUS --learning_rate_factor $LEARNING_RATE_FACTOR $FROZEN_ENCODER_FLAG $RANDINIT_FLAG
# EOL
# 		    	qsub "$PBS_SCRIPT"
# 			done
# 	    done
#     done
# done

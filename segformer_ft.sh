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
model="convnext"
# pretrain_scheme="simclr"
# randinit_args=("" "--rand_init")
# randinit_args=("--rand_init")
# for randinit_arg in "${randinit_args[@]}"
pretrain_schems=( "hires_simclr" "simclr")
# pretrain_schems=("simclr")
randinit_arg=""
for pretrain_scheme in "${pretrain_schems[@]}"
do
    MINI_BATCH_SIZE=256
    FULL_BATCH_SIZE=256
    if [[ "$pretrain_scheme" == "hires_simclr" ]]; then
        IMAGE_SIZE=192
    else
        IMAGE_SIZE=256
    fi
    # IMAGE_SIZE=192
    LEARNING_RATE=0.0001
    EARLY_STOPPING=3
    MAX_EPOCHS=100

    WEIGHTS_DIR='./weights/'
    JOB_NAME="${NAME}_${model}_${pretrain_scheme}"
    PROG_LOG_DIR='./logs/convnext_tests'
    mkdir -p "$PROG_LOG_DIR"
    mkdir -p "$PROG_LOG_DIR/$model"

    python $SCRIPT_NAME --pretrain_scheme $pretrain_scheme --model $model --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --weights_dir $WEIGHTS_DIR --log_dir $PROG_LOG_DIR --image_size $IMAGE_SIZE --init_lr $LEARNING_RATE --early_stopping_patience $EARLY_STOPPING --num_epochs $MAX_EPOCHS $randinit_arg

done

HIRES_SIMCLR_WEIGHTS_PATH="./weights/convnext/hires_simclr.pth"
SIMCLR_WEIGHTS_PATH="./weights/convnext/simclr.pth"
# SIMCLR_RANDINIT_WEIGHTS_PATH="./weights/simclr2a_tests/resnet152d_randinit/simclr.pth"

# Ensure the log directory exists
# mkdir -p "$PBS_LOG_DIR"

pretrain_schemes=('simclr' 'imagenet' 'randinit')
# pretrain_schemes=('hires_simclr')
# ft_datas=("./data/splits" "./data/cpb_tests/splits")
ft_datas=("./data/splits")

freeze_encoders=("" "--freeze_encoder")

# source $PYTHON_ENV
for ft_data in "${ft_datas[@]}"; do
    for pretrain_scheme in "${pretrain_schemes[@]}"; do
        for freeze_encoder in "${freeze_encoders[@]}"; do

            if [[ "$pretrain_scheme" == "randinit" && "$freeze_encoder" == "--freeze_encoder" ]]; then
                continue
            fi

            SUB_DIR="msft1_convnext/${pretrain_scheme}"

            if [[ "$ft_data" == *"cpb_tests"* ]]; then
                SUB_DIR="${SUB_DIR}/cpb"
                BATCH_SIZE=16
            else
                SUB_DIR="${SUB_DIR}/mslc"
                BATCH_SIZE=8
            fi

            if [[ "$freeze_encoder" == "--freeze_encoder" ]]; then
                SUB_DIR="${SUB_DIR}/decoder_train"
            else
                SUB_DIR="${SUB_DIR}/full_train"
            fi

            LOG_DIR="./logs/${SUB_DIR}"
            OUT_DIR="./weights/${SUB_DIR}"

            # if classification_report.csv already exists in log_dir, skip
            # if [[ -f "${LOG_DIR}/classification_report.csv" ]]; then
                # continue
            # fi

            JOB_NAME="resunet_${pretrain_scheme}_ft_stage1${freeze_encoder}"
            if [[ "$ft_data" == *"cpb_tests"* ]]; then
                JOB_NAME="${JOB_NAME}_cpb"
            else
                JOB_NAME="${JOB_NAME}_mslc"
            fi
                                    
            # Create a PBS script for the job
            PBS_SCRIPT="./pbs_scripts/${JOB_NAME}.pbs"
            mkdir -p "$(dirname "$PBS_SCRIPT")"

            if [[ "$pretrain_scheme" == "imagenet" ]]; then
                WEIGHTS_PARAMETER="--encoder_weights imagenet"
            elif [[ "$pretrain_scheme" == "simclr" ]]; then
                WEIGHTS_PARAMETER="--encoder_weights ${SIMCLR_WEIGHTS_PATH}"
            elif [[ "$pretrain_scheme" == "simclr_randinit" ]]; then
                WEIGHTS_PARAMETER="--encoder_weights ${SIMCLR_RANDINIT_WEIGHTS_PATH}"
            elif [[ "$pretrain_scheme" == "hires_simclr" ]]; then
                WEIGHTS_PARAMETER="--encoder_weights ${HIRES_SIMCLR_WEIGHTS_PATH}"
            else
                WEIGHTS_PARAMETER=""
            fi

            TRAIN_DIR="${ft_data}/train/"
            VAL_DIR="${ft_data}/val/"
            TEST_DIR="${ft_data}/test/"
            
            arguments=(
                "finetune.py"
                "--model" "convnext"
                "--train_dir" "$TRAIN_DIR"
                "--val_dir" "$VAL_DIR"
                "--test_dir" "$TEST_DIR"
                "--log_dir" "$LOG_DIR"
                "--output_dir" "$OUT_DIR"
                "--mini_batch_size" "$BATCH_SIZE"
                "--full_batch_size" "$BATCH_SIZE"
                "--load_data_from_disk"
                "--lr" "0.0001" # default is 1e-5
                # "--num_workers" "$NCPUS"
                "$freeze_encoder"
            )

            # Split WEIGHTS_PARAMETER into separate elements
            IFS=' ' read -r -a weights_array <<< "$WEIGHTS_PARAMETER"
            arguments+=("${weights_array[@]}")

            # if running on GCER GPU server, run directly
            if [[ "$HOSTNAME" == "gcer-a100" ]]; then
                echo "python ${arguments[@]}"
                python "${arguments[@]}"
                continue
            else
                # add --num_workers to arguments
                arguments+=("--num_workers" "$NCPUS")
            fi
        done
    done
done

# # stage 2 - decoder pretrain
# model="att_unet"
# # encoder_weights=("simclr" "simclr_randinit" "imagenet" "randinit")
# encoder_weights=("simclr_randinit" "imagenet")
# encoder_freeze_flags=("--frozen_encoder")
# # decoder_strategies=("dae" "dae_si")
# decoder_strategies=("dae")
# for encoder_weights in "${encoder_weights[@]}"
# do
#     if [ "$encoder_weights" = "randinit" ]; then
#         weights_argument="--rand_init"
#     elif [ "$encoder_weights" = "simclr" ]; then
#         weights_argument="--encoder_weights ./weights/simclr2a_tests/resnet152d/simclr.pth"
#     elif [ "$encoder_weights" = "simclr_randinit" ]; then
#         weights_argument="--encoder_weights ./weights/simclr2a_tests/resnet152d_randinit/simclr.pth"
#     elif [ "$encoder_weights" = "imagenet" ]; then
#         weights_argument=""
#     fi

#     for encoder_freeze_flags in "${encoder_freeze_flags[@]}"
#     do

#         # skip if encoder is frozen and randinit
#         if [ "$encoder_weights" = "randinit" -a "$encoder_freeze_flags" = "--frozen_encoder" ]; then
#             continue
#         fi

#         for decoder_strategy in "${decoder_strategies[@]}"
#         do

#             MINI_BATCH_SIZE=8
#             FULL_BATCH_SIZE=8
#             IMAGE_SIZE=256
#             EARLY_STOPPING=3

#             WEIGHTS_DIR='./weights/simclr2a_tests'
#             PROG_LOG_DIR='./logs/simclr2a_tests_decoder'
#             mkdir -p "$PROG_LOG_DIR"
#             mkdir -p "$PROG_LOG_DIR/att_unet"

#             python $SCRIPT_NAME --pretrain_scheme $decoder_strategy --model att_unet --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --weights_dir $WEIGHTS_DIR --log_dir $PROG_LOG_DIR --image_size $IMAGE_SIZE --early_stopping_patience $EARLY_STOPPING $weights_argument $encoder_freeze_flags

#         done

#     done

# done

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

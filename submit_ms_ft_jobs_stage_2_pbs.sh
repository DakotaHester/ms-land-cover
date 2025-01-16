#!/bin/bash

# Set common PBS parameters
QUEUE="biggpu"
NCPUS="8"
NGPUS="1"
MEMORY="64GB"
TIME="240:00:00"
MAIL_USER="dh2306@msstate.edu"

SCRIPT_NAME="finetune_unet.py"
PBS_LOG_DIR="./logs/ms_ft_stage2/pbs"

# Ensure the log directory exists
mkdir -p "$PBS_LOG_DIR"

TRAIN_DIR="${FT_DATA}/train/"
VAL_DIR="${FT_DATA}/val/"
TEST_DIR="${FT_DATA}/test/"



pretrain_schemes=('dae' 'imagenet' 'randinit')
stage_1_frozen_encoders=('true' 'false')
freeze_encoders=("" "--freeze_encoder")
freeze_decoders=("" "--freeze_decoder")

# source $PYTHON_ENV

for pretrain_scheme in "${pretrain_schemes[@]}"
do
    for stage_1_frozen_encoder in "${stage_1_frozen_encoders[@]}"
    do

        # did not train with frozen encoder/randinit in stage 1
        if [ "$stage_1_frozen_encoder" = true -a $pretrain_scheme = "randinit" ]; then
            continue
        fi

        for freeze_encoder in "${freeze_encoders[@]}"
        do

            for freeze_decoder in "${freeze_decoders[@]}"
            do

                SUB_DIR="multistage_finetuning_stage2/${pretrain_scheme}"
                JOB_NAME="resunet_${pretrain_scheme}_ft_stage2"
                WEIGHTS_DIR="./weights/multistage_finetuning_stage1/${pretrain_scheme}_bak/cpb"
                if [ "$stage_1_frozen_encoder" = true ]; then
                    SUB_DIR="${SUB_DIR}/s1_frozen_encoder"
                    WEIGHTS_DIR="$WEIGHTS_DIR/cpb/decoder_train/best_model.pth"
                    JOB_NAME="${JOB_NAME}_s1_decoder_train"
                else
                    SUB_DIR="${SUB_DIR}/s1_full_encoder"
                    WEIGHTS_DIR="$WEIGHTS_DIR/cpb/full_encoder/best_model.pth"
                    JOB_NAME="${JOB_NAME}_s1_full_train"
                fi

                if [ "$freeze_encoder" = "--freeze_encoder" -a "$freeze_decoder" = "--freeze_decoder" ]; then
                    SUB_DIR="${SUB_DIR}/s2_linear_probe"
                    JOB_NAME="${JOB_NAME}_s2_linear_probe"
                elif [ "$freeze_encoder" = "--freeze_encoder" ]; then
                    SUB_DIR="${SUB_DIR}/s2_decoder_train"
                    JOB_NAME="${JOB_NAME}_s2_decoder_train"
                elif [ "$freeze_decoder" = "--freeze_decoder" ]; then
                    SUB_DIR="${SUB_DIR}/s2_encoder_train"
                    JOB_NAME="${JOB_NAME}_s2_encoder_train"
                else
                    SUB_DIR="${SUB_DIR}/s2_full_train"
                    JOB_NAME="${JOB_NAME}_s2_full_train"
                fi

                LOG_DIR="./logs/${SUB_DIR}"
                OUT_DIR="./weights/${SUB_DIR}"
                
                # if log_dir/test_metriics.csv exists, skip
                if [ -f "${LOG_DIR}/test_metrics.csv" ]; then
                    continue
                fi


                PBS_SCRIPT="./pbs_scripts/${JOB_NAME}.pbs"

                arguments=(
                    "finetune_unet.py"
                    "--train_dir" "$TRAIN_DIR"
                    "--val_dir" "$VAL_DIR"
                    "--test_dir" "$TEST_DIR"
                    "--weights_dir" "$WEIGHTS_DIR"
                    "--log_dir" "$LOG_DIR"
                    "--out_dir" "$OUT_DIR"
                    "--lr 1e-5"
                    $frozen_encoder
                )
        
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
module load cuda10.2/toolkit
module load python
conda init
source ~/.bashrc
conda activate mslc
export CUDA_VISIBLE_DEVICES=0
python ${arguments[@]}
EOL
        # Submit the job
                qsub "$PBS_SCRIPT"
            done
        done
    done
done

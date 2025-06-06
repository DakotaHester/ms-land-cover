#!/bin/bash
#
## PBS Params
NAME="unet-finetune"
QUEUE="biggpu"
NCPUS="8"
NGPUS="1"
MEMORY="32GB"
TIME="240:00:00"
MAIL_USER="dh2306@msstate.edu"

PYTHON_ENV=".env/bin/activate"
SCRIPT_NAME="finetune_unet.py"
FT_DATA="./data/splits"
PBS_LOG_DIR="./logs/pbs/finetune_unet"

mkdir -p "$PBS_LOG_DIR"

for model in unet
do
    for pretrain_scheme in std std_randinit std_fe imagenet randinit
    do
        for pretrain_task in dae lab dae_lab none
        do

            # if pretrain_scheme contains std, enter the if block
            if [[ "$pretrain_scheme" == *"std"* ]]; then
                if [[ "$pretrain_task" == "none" ]]; then
                    continue
                elif [[ "$pretrain_scheme" == "std" ]]; then
                    WEIGHTS_PATH="./weights/unet/${pretrain_task}.pth"
                elif [[ "$pretrain_scheme" == "std_randinit" ]]; then
                    WEIGHTS_PATH="./weights/unet_randinit/${pretrain_task}.pth"
                elif [[ "$pretrain_scheme" == "std_fe" ]]; then
                    WEIGHTS_PATH="./weights/unet_frozenencoder/${pretrain_task}.pth"
                fi
                WEIGHTS_ARG="--model_weights $WEIGHTS_PATH"
            else
                # if pretrain_task != none, skip (no pretraining)
                if [[ "$pretrain_task" != "none" ]]; then
                    continue
                elif [[ "$pretrain_scheme" == "imagenet" ]]; then
                    WEIGHTS_ARG="--encoder_weights imagenet"
                elif [[ "$pretrain_scheme" == "randinit" ]]; then
                    WEIGHTS_ARG=""
                fi
            fi

            for finetune_frozen_encoder in 0 1
            do

                # if finetune_frozen_encoder is 1 and pretrain_scheme is randinit, skip
                if [[ "$finetune_frozen_encoder" == "1" ]]; then
                    if [[ "$pretrain_scheme" == "randinit" ]]; then
                        continue
                    fi
                    FROZEN_ENCODER_FLAG="--freeze_encoder"
                else
                    FROZEN_ENCODER_FLAG=""
                fi


                for finetune_frozen_decoder in 0 1
                do
                    
                    # if finetune_frozen_decoder is 1 and pretrain_scheme does not contain std, skip
                    if [[ "$finetune_frozen_decoder" == "1" ]]; then
                        if [[ "$pretrain_scheme" != *"std"* ]]; then
                            continue
                        fi
                        FROZEN_DECODER_FLAG="--freeze_decoder"
                    else
                        FROZEN_DECODER_FLAG=""
                    fi
                    
                    JOB_NAME="${NAME}_${model}_${pretrain_scheme}_${pretrain_task}"
                    OUT_DIR="${model}/${pretrain_scheme}/${pretrain_task}"

                    # if finetune_frozen_encoder is 1 and finetune_frozen_decoder is 1, add linear_probe to out_path and job_name
                    if [[ "$finetune_frozen_encoder" == "1" && "$finetune_frozen_decoder" == "1" ]]; then
                        OUT_DIR="${OUT_DIR}/linear_probe"
                        JOB_NAME="${JOB_NAME}_linear_probe"
                    elif [[ "$finetune_frozen_encoder" == "1" ]]; then
                        OUT_DIR="${OUT_DIR}/frozen_encoder"
                        JOB_NAME="${JOB_NAME}_frozen_encoder"
                    elif [[ "$finetune_frozen_decoder" == "1" ]]; then
                        OUT_DIR="${OUT_DIR}/frozen_decoder"
                        JOB_NAME="${JOB_NAME}_frozen_decoder"
                    else
                        OUT_DIR="${OUT_DIR}/full_train"
                        JOB_NAME="${JOB_NAME}_full_train"
                    fi

                    TRAIN_DIR="${FT_DATA}/train/"
                    VAL_DIR="${FT_DATA}/val/"
                    TEST_DIR="${FT_DATA}/test/"

                    OUT_DIR="./weights/${OUT_DIR}"
                    LOG_DIR="./logs/${OUT_DIR}"

                    arguments=(
                        "finetune_unet.py"
                        "--train_dir" "$TRAIN_DIR"
                        "--val_dir" "$VAL_DIR"
                        "--test_dir" "$TEST_DIR"
                        "--output_dir" "$OUT_DIR"
                        "--log_dir" "$LOG_DIR"
                        "--num_workers" "$NCPUS"
                        "$WEIGHTS_ARG"
                        "$FROZEN_ENCODER_FLAG"
                        "$FROZEN_DECODER_FLAG"
                    )

                    PBS_SCRIPT="./pbs_scripts/${JOB_NAME}.pbs"
                    mkdir -p "$(dirname "$PBS_SCRIPT")"

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
# #PBS -m abe
# #PBS -M $MAIL_USER

cd \${PBS_O_WORKDIR}
module load cuda10.2/toolkit
module load python
conda init
source ~/.bashrc
conda activate mslc
python ${arguments[@]}
EOL
		    	    qsub "$PBS_SCRIPT"
                done
			done
        done
    done
done

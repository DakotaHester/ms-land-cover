#!/bin/bash

# Set common PBS parameters
QUEUE="biggpu"
NCPUS="1"
NGPUS="1"
MEMORY="16GB"
TIME="48:00:00"
MAIL_USER="dh2306@msstate.edu"

SCRIPT_NAME="finetune_unet.py"
# FT_DATA="./data/cpb_tests/splits"
DAE_SI_WEIGHTS_PATH="./weights/hrunet/dae_si.pth"

# Ensure the log directory exists
# mkdir -p "$PBS_LOG_DIR"

pretrain_schemes=('imagenet' 'dae_si' 'randinit')
# ft_datas=("./data/cpb_tests/splits" "./data/splits")
ft_datas=("./data/splits")

freeze_encoders=("" "--freeze_encoder" )
freeze_decoders=("" "--freeze_decoder")

# source $PYTHON_ENV
for ft_data in "${ft_datas[@]}"; do
    for pretrain_scheme in "${pretrain_schemes[@]}"; do
        for freeze_encoder in "${freeze_encoders[@]}"; do
            for freeze_decoder in "${freeze_decoders[@]}"; do
                if [[ "$pretrain_scheme" == "randinit" && "$freeze_encoder" == "--freeze_encoder" ]]; then
                    continue
                fi

                SUB_DIR="msft1_HRUNET/${pretrain_scheme}"

                if [[ "$ft_data" == *"cpb_tests"* ]]; then
                    SUB_DIR="${SUB_DIR}/cpb"
                else
                    SUB_DIR="${SUB_DIR}/mslc"
                fi

                # if [ "$freeze_encoder" = "--freeze_encoder" -a "$freeze_decoder" = "--freeze_decoder" ]; then
                #     SUB_DIR="${SUB_DIR}/s2_linear_probe"
                #     JOB_NAME="${JOB_NAME}_s2_linear_probe"
                # elif [ "$freeze_encoder" = "--freeze_encoder" ]; then
                #     SUB_DIR="${SUB_DIR}/s2_decoder_train"
                #     JOB_NAME="${JOB_NAME}_s2_decoder_train"
                # elif [ "$freeze_decoder" = "--freeze_decoder" ]; then
                #     SUB_DIR="${SUB_DIR}/s2_encoder_train"
                #     JOB_NAME="${JOB_NAME}_s2_encoder_train"
                # else
                #     SUB_DIR="${SUB_DIR}/s2_full_train"
                #     JOB_NAME="${JOB_NAME}_s2_full_train"
                # fi

                if [ "$freeze_encoder" = "--freeze_encoder" -a "$freeze_decoder" = "--freeze_decoder" ]; then
                    SUB_DIR="${SUB_DIR}/linear_probe"
                elif [[ "$freeze_encoder" == "--freeze_encoder" ]]; then
                    SUB_DIR="${SUB_DIR}/decoder_train"
                elif [[ "$freeze_decoder" == "--freeze_decoder" ]]; then
                    SUB_DIR="${SUB_DIR}/encoder_train"
                else
                    SUB_DIR="${SUB_DIR}/full_train"
                fi
                    
                # elif [[ "$freeze_encoder" == "--freeze_encoder" ]]; then
                #     SUB_DIR="${SUB_DIR}/decoder_train"
                # else
                #     SUB_DIR="${SUB_DIR}/full_train"
                # fi

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
                elif [[ "$pretrain_scheme" == "dae_si" ]]; then
                    WEIGHTS_PARAMETER="--model_weights ${DAE_SI_WEIGHTS_PATH}"
                else
                    WEIGHTS_PARAMETER=""
                fi

                TRAIN_DIR="${ft_data}/train/"
                VAL_DIR="${ft_data}/val/"
                TEST_DIR="${ft_data}/test/"
                
                arguments=(
                    "finetune_unet.py"
                    "--model" "hrunet"
                    "--train_dir" "$TRAIN_DIR"
                    "--val_dir" "$VAL_DIR"
                    "--test_dir" "$TEST_DIR"
                    "--log_dir" "$LOG_DIR"
                    "--output_dir" "$OUT_DIR"
                    # "--num_workers" "$NCPUS"
                    "$freeze_encoder"
                    "$freeze_decoder"
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
                
                cat > "$PBS_SCRIPT" <<EOL
#!/bin/bash
#PBS -N $JOB_NAME
#PBS -q $QUEUE
#PBS -j oe
#PBS -o $LOG_DIR/job.out
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEMORY
#PBS -l walltime=$TIME
##PBS -m abe
##PBS -M $MAIL_USER

cd \${PBS_O_WORKDIR}
module load cuda10.2/toolkit
module load python
conda init
source ~/.bashrc
conda activate mslc
export CUDA_VISIBLE_DEVICES=0
python ${arguments[@]} --load_data_from_disk
EOL
                # Submit the job
                timestamp=$(date -u +"[%Y-%m-%d %H:%M:%SZ]")
                echo "=========================="
                echo "$timestamp Submitting job: $JOB_NAME"
                echo "$timestamp Arguments: ${arguments[@]}"
                echo "$timestamp PBS script: $PBS_SCRIPT"
                qsub "$PBS_SCRIPT"
                echo "==========================="
            done
        done
    done
done

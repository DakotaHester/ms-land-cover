#!/bin/bash

# Set common PBS parameters
QUEUE="biggpu"
NCPUS="1"
NGPUS="1"
MEMORY="8GB"
TIME="48:00:00"
MAIL_USER="dh2306@msstate.edu"

SCRIPT_NAME="finetune_unet.py"
PBS_LOG_DIR="./logs/ms_ft_stage1_202502/pbs"
# FT_DATA="./data/cpb_tests/splits"
DAE_WEIGHTS_PATH="./weights/resnet152/dae.pth"

# Ensure the log directory exists
mkdir -p "$PBS_LOG_DIR"

pretrain_schemes=('randinit' 'imagenet' 'dae')
ft_datas=("./data/splits" "./data/cpb_tests/splits")

freeze_encoders=("--freeze_encoder" "")

# source $PYTHON_ENV
for pretrain_scheme in "${pretrain_schemes[@]}"; do
    for freeze_encoder in "${freeze_encoders[@]}"; do
        for ft_data in "${ft_datas[@]}"; do

            if [[ "$pretrain_scheme" == "randinit" && "$freeze_encoder" == "--freeze_encoder" ]]; then
                continue
            fi

            SUB_DIR="msft1_202502/${pretrain_scheme}"

            if [[ "$ft_data" == *"cpb_tests"* ]]; then
                SUB_DIR="${SUB_DIR}/cpb"
            else
                SUB_DIR="${SUB_DIR}/mslc"
            fi

            if [[ "$freeze_encoder" == "--freeze_encoder" ]]; then
                SUB_DIR="${SUB_DIR}/decoder_train"
            else
                SUB_DIR="${SUB_DIR}/full_train"
            fi

            LOG_DIR="./logs/${SUB_DIR}"
            OUT_DIR="./weights/${SUB_DIR}"

            # if classification_report.csv already exists in log_dir, skip
            if [[ -f "${LOG_DIR}/classification_report.csv" ]]; then
                continue
            fi

            JOB_NAME="resunet_${pretrain_scheme}_ft_stage1${freeze_encoder}_{$ft_data##*/}"
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
            elif [[ "$pretrain_scheme" == "dae" ]]; then
                WEIGHTS_PARAMETER="--encoder_weights ${DAE_WEIGHTS_PATH}"
            else
                WEIGHTS_PARAMETER=""
            fi

            TRAIN_DIR="${ft_data}/train/"
            VAL_DIR="${ft_data}/val/"
            TEST_DIR="${ft_data}/test/"
            
            arguments=(
                "finetune_unet.py"
                "--train_dir" "$TRAIN_DIR"
                "--val_dir" "$VAL_DIR"
                "--test_dir" "$TEST_DIR"
                "$WEIGHTS_PARAMETER"
                "--log_dir" "$LOG_DIR"
                "--output_dir" "$OUT_DIR"
                "--num_workers" "$NCPUS"
                "$freeze_encoder"
                "--load_data_from_disk" # save RAM
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
##PBS -m abe
##PBS -M $MAIL_USER

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

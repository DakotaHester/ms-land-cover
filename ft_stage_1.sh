#!/bin/bash
DAE_WEIGHTS_PATH = 

for pretrain_scheme in imagenet dae randinit
do
    for ft_data in "./data/splits" "./data/cpb_tests/splits"
    do
        for frozen_encoder in "" "--frozen_encoder"
        do

            if [[ $pretrain_scheme == "randinit" && $frozen_encoder == "--frozen_encoder" ]]; then
                continue
            fi

            TRAIN_DIR="${ft_data}/train/"
            VAL_DIR="${ft_data}/val/"
            TEST_DIR="${ft_data}/test/"

            SUB_DIR="multistage_finetuning_stage1/${pretrain_scheme}"

            if [[ $ft_data == *"cpb_tests"* ]]; then
                SUB_DIR="${SUB_DIR}/cpb"
            else
                SUB_DIR="${SUB_DIR}/mslc"
            fi

            if [ "$frozen_encoder" == "--frozen_encoder" ]; then
                SUB_DIR="${SUB_DIR}/decoder_train"
            else
                SUB_DIR="${SUB_DIR}/full_train"
            fi

            LOG_DIR="./logs/${SUB_DIR}"
            OUT_DIR="./weights/${SUB_DIR}"

            python finetune_unet.py \
                --pretrain_scheme $pretrain_scheme \
                --train_dir $TRAIN_DIR \
                --val_dir $VAL_DIR \
                --test_dir $TEST_DIR \
                --log_dir $LOG_DIR \
                --output_dir $OUT_DIR \
                $frozen_encoder

            echo "==="
            echo $LOG_DIR
            echo $OUT_DIR
            echo "==="
        
        done
    done
done

            # if [[ "$ft_data" == "./data/splits" && "$frozen_encoder" == "--frozen_encoder" ]]; then
            #     continue
            # fi
            # LOG_DIR = "./logs/cpb_tests/ft"
            # if [ "$frozen_encoder" = "--frozen_encoder" ]; then
            #     JOB_NAME="dh2306_${pretrain_scheme}_ft_simple_decoder_test"
            # else
            #     JOB_NAME="dh2306_${pretrain_scheme}_ft_test"
            # fi
            # SLURM_SCRIPT="./slurm_scripts/${JOB_NAME}.slurm"
            # mkdir -p "$(dirname "$SLURM_SCRIPT")"
            # cat > "$SLURM_SCRIPT" <<EOL
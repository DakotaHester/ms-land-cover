#!/bin/bash

for pretrain_scheme in imagenet dae randinit
do
    for stage_1_frozen_encoder in true false
    do

        # did not train with frozen encoder/randinit in stage 1
        if [ "$stage_1_frozen_encoder" = true && $pretrain_scheme == "randinit" ]; then
            continue
        fi

        for frozen_encoder in "" "--frozen_encoder"
        do

            for frozen_decoder in "" "--frozen_decoder"
            do

            # skip training encoder with frozen decoder
            if [ "$frozen_encoder" = "" & "$frozen_decoder" = "--frozen_decoder" ]; then
                continue
            fi

            SUB_DIR="multistage_finetuning_stage2/${pretrain_scheme}"

            if [ "$stage_1_frozen_encoder" = true ]; then
                SUB_DIR="${SUB_DIR}/s1_frozen_encoder"
            else
                SUB_DIR="${SUB_DIR}/s1_full_encoder"
            fi

            if [ "$frozen_encoder" = "--frozen_encoder" ]; then
                SUB_DIR="${SUB_DIR}/s2_frozen_encoder"
            else
                SUB_DIR="${SUB_DIR}/s2_full_encoder"
            fi

            if [ "$frozen_decoder" = "--frozen_decoder" ]; then
                SUB_DIR="${SUB_DIR}/s2_frozen_decoder"
            else
                SUB_DIR="${SUB_DIR}/s2_full_decoder"
            fi

            LOG_DIR="./logs/${SUB_DIR}"
            OUT_DIR="./weights/${SUB_DIR}"
        
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
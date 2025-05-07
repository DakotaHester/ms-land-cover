HIRES_SIMCLR_WEIGHTS_PATH="./weights/convnext/hires_simclr.pth"
SIMCLR_WEIGHTS_PATH="./weights/convnext/simclr.pth"

# Hyperparameters
MODEL="convnext"
MINI_BATCH_SIZE=8
FULL_BATCH_SIZE=8
LR=0.00001
NUM_EPOCHS=1000
FOCAL_GAMMA=2.0

# pretrain_schemes=('dae_si' 'dae' 'none')
pretrain_schemes=('dae' 'dae_si' 'none')
freeze_encoders=("frozen_encoder" "no_freeze")
all_init_encoder_weights=("hires_simclr" "simclr" "randinit" "imagenet")
finetune_weights_trained=("full_model" "decoder_only" "linear_probe")
for freeze_encoders in ${freeze_encoders[@]}
do
    for pretrain_scheme in ${pretrain_schemes[@]}
    do
        for init_encoder_weights in ${all_init_encoder_weights[@]}
        do
            
            if [[ "$init_encoder_weights" == "randinit" && "$freeze_encoders" == "--frozen_encoder" ]]; then
                continue
            fi

            for finetune_training in ${finetune_weights_trained[@]}
            do

                weights_trained_arg=""
                if [[ "$finetune_training" == "decoder_only" ]]; then
                    weights_trained_arg="--freeze_encoder"
                elif [[ "$finetune_training" == "linear_probe" ]]; then
                    weights_trained_arg="--freeze_encoder --freeze_decoder"
                fi
                JOB_NAME="${NAME}_${model}_${pretrain_scheme}_${freeze_encoders}_${init_encoder_weights}_${finetune_training}"
                OUT_DIR="${model}/${pretrain_scheme}/${freeze_encoders}/${init_encoder_weights}/${finetune_training}"

                if [[ ${pretrain_scheme} != "none" ]]; then

                    WEIGHTS_ARG="--model_weights ./weights/att_unext_pts2/${init_encoder_weights}/att_unext"

                    # if [[ "$init_encoder_weights" == "simclr" ]]; then
                    #     WEIGHTS_ARG="--model_weights ./weights/att_unext_pts2/convnext/simclr/att_unext/${pretrain_scheme}"
                    # elif [[ "$init_encoder_weights" == "hires_simclr" ]]; then
                    #     WEIGHTS_ARG="--model_weights ./weights/att_unext_pts2/convnext/hires_simclr/att_unext/${pretrain_scheme}"
                    # elif [[ "$init_encoder_weights" == "randinit" ]]; then
                    #     WEIGHTS_ARG="--model_weights ./weights/att_unext_pts2/randinit/att_unext/${pretrain_scheme}"
                    # elif [[ "$init_encoder_weights" == "imagenet" ]]; then
                    #     WEIGHTS_ARG="--model_weights ./weights/att_unext_pts2/imagenet/att_unext/${pretrain_scheme}"
                    # fi

                    if [[ "$freeze_encoders" == "frozen_encoder" ]]; then
                        WEIGHTS_ARG="${WEIGHTS_ARG}_frozenencoder"
                    fi
                    WEIGHTS_ARG="${WEIGHTS_ARG}/${pretrain_scheme}.pth"
                
                elif [[ "$pretrain_scheme" == "none" ]]; then
                    if [[ "$init_encoder_weights" == "imagenet" ]]; then
                        WEIGHTS_ARG="--encoder_weights imagenet"
                    elif [[ "$init_encoder_weights" == "randinit" ]]; then
                        WEIGHTS_ARG=""
                    elif [[ "$init_encoder_weights" == "simclr" ]]; then
                        WEIGHTS_ARG="--encoder_weights ${SIMCLR_WEIGHTS_PATH}"
                    elif [[ "$init_encoder_weights" == "hires_simclr" ]]; then
                        WEIGHTS_ARG="--encoder_weights ${HIRES_SIMCLR_WEIGHTS_PATH}"
                    fi
                fi

                # OUTPUT_DIR='./weights/hires_simclr_tests_finetune/'${init_encoder_weights}'/'${finetune_training}'/'${freeze_encoders}'/'${pretrain_scheme}
                OUTPUT_DIR="./weights/hires_simclr_tests_finetune2/${init_encoder_weights}/${finetune_training}/${freeze_encoders}/${pretrain_scheme}"
                LOG_DIR="./logs/hires_simclr_tests_finetune2/${init_encoder_weights}/${finetune_training}/${freeze_encoders}/${pretrain_scheme}"

                # if directories do not exist, then create them
                mkdir -p "$OUTPUT_DIR"
                mkdir -p "$LOG_DIR"

                # if LOG_DIR/classification_report.csv exists, then skip this run
                # if [[ -f "$LOG_DIR/classification_report.csv" ]]; then
                    # echo "Skipping run for $JOB_NAME as classification_report.csv already exists."
                    # continue
                # fi


                mkdir -p "$LOG_DIR"
                mkdir -p "$LOG_DIR/$model"
                echo "python finetune.py --model $MODEL --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --focal_gamma $FOCAL_GAMMA --log_dir $LOG_DIR --output_dir $OUTPUT_DIR --lr $LR --num_epochs $NUM_EPOCHS ${WEIGHTS_ARG} ${weights_trained_arg}"
                python finetune.py --model $MODEL --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --focal_gamma $FOCAL_GAMMA --log_dir $LOG_DIR --output_dir $OUTPUT_DIR --lr $LR --num_epochs $NUM_EPOCHS ${WEIGHTS_ARG} ${weights_trained_arg}

            # if [[ "$freeze_encoders" == "no_freeze" ]]; then
            #     freeze_encoders=""
            # fi
            
            # if [[ "$init_encoder_weights" == "simclr" ]]; then
            #     weights_arg="--encoder_weights ${SIMCLR_WEIGHTS_PATH}"
            # elif [[ "$init_encoder_weights" == "hires_simclr" ]]; then
            #     weights_arg="--encoder_weights ${HIRES_SIMCLR_WEIGHTS_PATH}"
            # elif [[ "$init_encoder_weights" == "randinit" ]]; then
            #     weights_arg="--rand_init"
            # elif [[ "$init_encoder_weights" == "imagenet" ]]; then
            #     weights_arg=""
            # fi

            # # if [[ "$SOME_CONDITION" == "true" ]]; then
            # #     # Add your logic here
            # # fi

            # MINI_BATCH_SIZE=64
            # FULL_BATCH_SIZE=64
            # IMAGE_SIZE=256
            # LEARNING_RATE=0.00001
            # EARLY_STOPPING=3
            # MAX_EPOCHS=100

            # WEIGHTS_DIR="./weights/att_unext_pts2/${init_encoder_weights}"

            # # create the weights directory if it doesn't exist
            # mkdir -p "$WEIGHTS_DIR"
            # # copy pretrain_mean and pretrain_std to the weights directory
            # cp ./weights/pretrain_mean.pth "$WEIGHTS_DIR"
            # cp ./weights/pretrain_std.pth "$WEIGHTS_DIR"

            # JOB_NAME="${NAME}_${model}_${pretrain_scheme}_${freeze_encoders}_${init_encoder_weights}"
            # PROG_LOG_DIR="./logs/hires_simclr_tests/${init_encoder_weights}"
            # mkdir -p "$PROG_LOG_DIR"
            # mkdir -p "$PROG_LOG_DIR/$model"

            # echo "python pretrain.py --model att_unext --pretrain_scheme $pretrain_scheme --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --weights_dir $WEIGHTS_DIR --log_dir $PROG_LOG_DIR --image_size $IMAGE_SIZE --init_lr $LEARNING_RATE --early_stopping_patience $EARLY_STOPPING --num_epochs $MAX_EPOCHS  ${weights_arg} ${freeze_encoders}"
            # python pretrain.py --model att_unext --pretrain_scheme $pretrain_scheme --mini_batch_size $MINI_BATCH_SIZE --full_batch_size $FULL_BATCH_SIZE --weights_dir $WEIGHTS_DIR --log_dir $PROG_LOG_DIR --image_size $IMAGE_SIZE --init_lr $LEARNING_RATE --early_stopping_patience $EARLY_STOPPING --num_epochs $MAX_EPOCHS ${weights_arg} ${freeze_encoders}
            done
        done
    done
done

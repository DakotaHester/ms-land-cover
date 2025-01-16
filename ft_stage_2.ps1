#!/usr/bin/env pwsh
$PYTHON_ENV = "./.env/Scripts/activate.ps1"
$TRAIN_DIR = "./data/splits/train"
$VAL_DIR = "./data/splits/val"
$TEST_DIR = "./data/splits/test"

$pretrain_schemes = @("imagenet", "dae", "randinit")
$stage_1_frozen_encoders = @($true, $false)
$freeze_encoders = @("", "--freeze_encoder")
$freeze_decoders = @("", "--freeze_decoder")

# activate python environment
$PYTHON_ENV

foreach ($pretrain_scheme in $pretrain_schemes) {
    foreach ($stage_1_frozen_encoder in $stage_1_frozen_encoders) {

        # did not train with frozen encoder/randinit in stage 1
        if ($stage_1_frozen_encoder -eq $true -and $pretrain_scheme -eq "randinit") {
            continue
        }

        foreach ($freeze_encoder in $freeze_encoders) {
            foreach ($freeze_decoder in $freeze_decoders) {

                $SUB_DIR = "multistage_finetuning_stage2/$pretrain_scheme"

                if ($stage_1_frozen_encoder -eq $true) {
                    $SUB_DIR = "$SUB_DIR/s1_frozen_encoder"
                    $WEIGHTS_DIR = "./weights/multistage_finetuning_stage1/$pretrain_scheme/cpb/decoder_train/best_model.pth"
                } else {
                    $SUB_DIR = "$SUB_DIR/s1_full_encoder"
                    $WEIGHTS_DIR = "./weights/multistage_finetuning_stage1/$pretrain_scheme/cpb/full_encoder/best_model.pth"
                }

                if ($freeze_encoder -eq "--freeze_encoder" -and $freeze_decoder -eq "--freeze_decoder") {
                    $SUB_DIR = "$SUB_DIR/s2_linear_probe"
                } elseif ($freeze_encoder -eq "--freeze_encoder") {
                    $SUB_DIR = "$SUB_DIR/s2_decoder_train"
                } elseif ($freeze_decoder -eq "--freeze_decoder") {
                    $SUB_DIR = "$SUB_DIR/s2_encoder_train"
                } else {
                    $SUB_DIR = "$SUB_DIR/s2_full_train"
                }

                $LOG_DIR = "./logs/$SUB_DIR"
                $OUT_DIR = "./weights/$SUB_DIR"
                
                # if log_dir/test_metrics.csv exists, skip
                if (Test-Path "$LOG_DIR/test_metrics.csv") {
                    continue
                }

                $arguments = @(
                    "finetune_unet.py"
                    "--train_dir", $TRAIN_DIR
                    "--val_dir", $VAL_DIR
                    "--test_dir", $TEST_DIR
                    "--model_weights", $WEIGHTS_DIR
                    "--log_dir", $LOG_DIR
                    "--output_dir", $OUT_DIR
                    $freeze_encoder
                    $freeze_decoder
                )

                Start-Process python -ArgumentList $arguments -NoNewWindow -Wait
            }
        }
    }
}

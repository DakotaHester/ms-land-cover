#!/usr/bin/env pwsh
$DAE_WEIGHTS_PATH = './weights/resnet152/dae.pth'
$PYTHON_ENV = './.env/Scripts/activate.ps1'

$pretrain_schemes = @('imagenet', 'dae', 'randinit')
$ft_datas = @("./data/splits", "./data/cpb_tests/splits")
$freeze_encoders = @("", "--freeze_encoder")

$PYTHON_ENV

foreach ($pretrain_scheme in $pretrain_schemes) {
    foreach ($ft_data in $ft_datas) {
        foreach ($freeze_encoder in $freeze_encoders) {

            if ($pretrain_scheme -eq "randinit" -and $freeze_encoder -eq "--freeze_encoder") {
                continue
            }

            $TRAIN_DIR = "${ft_data}/train/"
            $VAL_DIR = "${ft_data}/val/"
            $TEST_DIR = "${ft_data}/test/"

            $SUB_DIR = "multistage_finetuning_stage1/${pretrain_scheme}"

            if ($ft_data -like "*cpb_tests*") {
                $SUB_DIR = "${SUB_DIR}/cpb"
            } else {
                $SUB_DIR = "${SUB_DIR}/mslc"
            }

            if ($freeze_encoder -eq "--freeze_encoder") {
                $SUB_DIR = "${SUB_DIR}/decoder_train"
            } else {
                $SUB_DIR = "${SUB_DIR}/full_train"
            }

            $LOG_DIR = "./logs/${SUB_DIR}"
            $OUT_DIR = "./weights/${SUB_DIR}"

            if ($pretrain_scheme -eq "imagenet") {
                $WEIGHTS_PARAMETER = "--encoder_weights imagenet"
            } elseif ($pretrain_scheme -eq "dae") {
                $WEIGHTS_PARAMETER = "--encoder_weights ${DAE_WEIGHTS_PATH}"
            } else {
                $WEIGHTS_PARAMETER = ""
            }
            
            $arguments = @(
                "finetune_unet.py"
                "--train_dir $TRAIN_DIR"
                "--val_dir $VAL_DIR"
                "--test_dir $TEST_DIR"
                $WEIGHTS_PARAMETER
                "--log_dir $LOG_DIR"
                "--output_dir $OUT_DIR"
                $freeze_encoder
            )
            Start-Process python -ArgumentList $arguments -NoNewWindow -Wait
        }
    }
}

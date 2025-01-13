
$models = @("resnet152")
$schemes = @("ae", "dae", "hsv", "dae_hsv", "simclr", "dae_simclr", "hsv_simclr", "dae_hsv_simclr", "ae_simclr")

foreach ($model in $models) {
    foreach ($scheme in $schemes) {
        python pretrain.py --model $model --pretrain_scheme $scheme --pretrain_hdf5_path .\data\splits\pretrain.hdf5 --num_epochs 10 --image_size 48 --mini_batch_size 128
    }
}
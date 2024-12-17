$models = @("hrnet_w18")
$schemes = @("simclr", "dae", "simclr_dae")
# $weights = @("--use_imagenet_weights", "")

foreach ($model in $models) {
    foreach ($scheme in $schemes) {
        python pretrain.py --model $model --pretrain_scheme $scheme --pretrain_hdf5_path .\data\splits\pretrain.hdf5
    }
}
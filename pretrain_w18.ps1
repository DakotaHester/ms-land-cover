$models = @("hrnet_w18")
$schemes = @("hsv", "simclr")
$weights = @("--use_imagenet_weights", "")

foreach ($model in $models) {
    foreach ($scheme in $schemes) {
        foreach ($weight in $weights) {
            python pretrain.py --model $model --pretrain_scheme $scheme --pretrain_hdf5_path .\data\splits\pretrain.hdf5 $weight
        }
    }
}
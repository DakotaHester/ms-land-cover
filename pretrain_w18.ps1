
$models = @("hrnet_w18")
$schemes = @("simclr", "dae_simclr", "hsv_simclr", "dae_hsv_simclr" "ae_simclr")

foreach ($model in $models) {
    foreach ($scheme in $schemes) {
        python pretrain.py --model $model --pretrain_scheme $scheme --pretrain_hdf5_path .\data\splits\pretrain.hdf5
    }
}

$models = @("resnet152")
$schemes = @("dae", "hsv", "dae_hsv", "ae")

foreach ($model in $models) {
    foreach ($scheme in $schemes) {
        python pretrain.py --model $model --pretrain_scheme $scheme --pretrain_hdf5_path .\data\splits\pretrain.hdf5 --num_epochs 5 --image_size 96 --mini_batch_size 32
    }
}
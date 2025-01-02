for model in hrnet_w18 hrnet_w48
do
    for pretrain_scheme in hsv dae dae_hsv
    do
        python pretrain.py --model $model --pretrain_scheme $pretrain_scheme --pretrain_hdf5_path ./pretrain.hdf5
    done
done
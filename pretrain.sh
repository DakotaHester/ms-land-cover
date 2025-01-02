for model in hrnet_w48
do
    for pretrain_scheme in simclr dae_simclr hsv_simclr dae_hsv_simclr
    do
        python pretrain.py --model $model --pretrain_scheme $pretrain_scheme
    done
done
for model in hrnet_w48
do
    for pretrain_scheme in simclr dae simclr_dae
    do
        python pretrain.py --model $model --pretrain_scheme $pretrain_scheme
    done
done
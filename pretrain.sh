for model in hrnet_w18 hrnet_w48
do
    for pretrain_scheme in hsv_simclr hsv simclr
    do
        python pretrain.py --model $model --pretrain_scheme $pretrain_scheme 
    done
done
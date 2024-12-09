for model in hrnet_w18 hrnet_w48
do
    for pretrain_scheme in hsv_simclr hsv simclr
    do
        for weights in '--use_imagenet_weights' ''
        do
            python pretrain.py --model $model --pretrain_scheme $pretrain_scheme $weights
        done
        # python pretrainpy --model $model --pretrain_scheme $pretrain_scheme 
    done
done
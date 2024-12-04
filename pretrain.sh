for model in hrnet_w18 hrnet_w48
do
    for pretrain_scheme in hsv_simclr hsv simclr
    do
        mkdir -p ./logs/pretrain/$pretrain_scheme/$model
        python pretrain.py --pretrain_scheme $pretrain_scheme --model $model | tee ./logs/pretrain/$pretrain_scheme/$model/log.txt
    done
done
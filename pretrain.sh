for model in hrnet_w18 hrnet_w48
do
    for pretrain_scheme in hsv_simclr hsv simclr
    do
        OUT_DIR=./logs/pretrain/$model/$pretrain_scheme
        mkdir $OUT_DIR
        python pretrain.py --model $model --pretrain_scheme $pretrain_scheme | tee $OUT_DIR/log.txt
    done
done
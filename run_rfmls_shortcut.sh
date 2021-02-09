export CUDA_VISIBLE_DEVICES=0
#!/bin/bash
exp='1C_wifi_raw_resnet_progressive'
echo $exp
echo "Start JOB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
savepath=''

python -u main.py \
    --exp_name $exp \
    --base_path *datapath* \
    --save_path $savepath \
    --file_type mat \
    --slice_size 512 \
    --devices 50 \
    --generator new \
    --add_padding True \
    --K 16 \
    --training_strategy big \
    --preprocessor no \
    --normalize True \
    --train False \
       --load-model *pre-trained model path* \
       --arch resnet \
       --depth 50 \
       --batch-size 128 \
       --multi-gpu \
       --no-tricks \
       --sparsity-type column \
       --admm \
       --epoch 7 \
       --admm-epochs 3 \
       --optmzr adam \
       --rho 0.01 \
       --rho-num 3 \
       --lr 0.0001 \
       --lr-scheduler cosine \
       --warmup \
       --warmup-epochs 5 \
       --mixup \
       --alpha 0.3 \
       --smooth \
       --smooth-eps 0.1 \
       --config-file config_res50_full_v2 \
       >$savepath/$exp/log.out &&
echo "Congratus! Finished *c4pattern* admm training!"&&
python -u main.py \
    --exp_name $exp \
    --base_path *datapath* \
    --save_path $savepath \
    --file_type mat \
    --slice_size 512 \
    --devices 50 \
    --generator new \
    --add_padding True \
    --K 16 \
    --training_strategy big \
    --preprocessor no \
    --normalize True \
    --train False \
       --load-model '' \
       --arch resnet \
       --depth 50 \
       --batch-size 128 \
       --multi-gpu \
       --no-tricks \
       --sparsity-type column \
       --masked-retrain \
       --epoch 10 \
       --optmzr adam \
       --lr 0.0001 \
       --lr-scheduler cosine \
       --warmup \
       --warmup-epochs 5 \
       --mixup \
       --alpha 0.3 \
       --smooth \
       --smooth-eps 0.1 \
       --config-file config_res50_full_v2 \
       >$savepath/$exp/log_masked.out &&
echo "Congratus! Finished masked retraining"
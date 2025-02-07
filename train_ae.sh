#!/bin/bash

loss="l2"
window_size=10
z_dim=128
encoding_noise=0.01
l2_norm_reg_enc=0.00001
num_layers=6
num_head=8
eps=1e-12
batch_size=1024
ae_epochs=15
num_workers=32
lr=1e-4
dropout=0.1
accum_grad_batches=1
warmup_steps=0
scheduler_steps=0
weight_decay=0.0
beta1=0.9
beta2=0.999
save_dir="/raid/monsals/mpdr_physics3"
name="ae-aug-z128-15e-bkgbbh"
log_every_n_steps=100
save_top_k=1
checkpoint_path="/raid/monsals/mpdr_physics3/checkpoints"
checkpoint_name="ae-aug-z128-15e-bkgbbh"
gpus=(2 3)

python -m train.ae \
    --latent_token \
    --augment \
    --pos_embedding \
    --loss $loss \
    --window_size $window_size \
    --z_dim $z_dim \
    --num_layers $num_layers \
    --num_head $num_head \
    --eps $eps \
    --batch_size $batch_size \
    --ae_epochs $ae_epochs \
    --num_workers $num_workers \
    --lr $lr \
    --dropout $dropout \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
    --scheduler_steps $scheduler_steps \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --gpus ${gpus[@]}


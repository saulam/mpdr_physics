#!/bin/bash

loss="l2"
window_size=10
z_dim=128
encoding_noise=0.01
l2_norm_reg_enc=0.00001
temperature=1
temperature_omi=1
gamma_vx=1  # hidden
gamma_neg_recon=1  # hidden
sampling_x=langevin
mcmc_n_step_x=11
mcmc_stepsize_x=4
mcmc_noise_x=0.130240
mcmc_n_step_omi=4
mcmc_stepsize_omi=0.400343
mcmc_noise_omi=0.150688
proj_mode=uniform
proj_noise_start=0.05
proj_noise_end=0.3
proj_const_omi=0.0001
proj_const=0.0001
proj_dist=geodesic  # hidden
num_layers=6
num_head=8
eps=1e-12
batch_size=512
nae_epochs=100
num_workers=32
lr=0.000407
dropout=0.1
accum_grad_batches=1
warmup_steps=5
scheduler_steps=0
weight_decay=0
beta1=0.9
beta2=0.999
save_dir="/raid/monsals/mpdr_physics3"
name="mpdr-r-z128-optuna3"
log_every_n_steps=10
save_top_k=1
pretrained_ae="/raid/monsals/mpdr_physics3/checkpoints/ae-aug-z128-15e-bkgbbh/last.ckpt"
pretrained_net_x="/raid/monsals/mpdr_physics3/checkpoints/ae-aug-z128-15e-bkgbbh/last.ckpt"
checkpoint_path="/raid/monsals/mpdr_physics3/checkpoints"
checkpoint_name="mpdr-r-z128-optuna3"
gpus=(6 7)

python -m train.mpdr-r \
    --latent_token \
    --pos_embedding \
    --learn_out_scale \
    --augment \
    --loss $loss \
    --window_size $window_size \
    --z_dim $z_dim \
    --temperature $temperature \
    --temperature_omi $temperature_omi \
    --sampling_x $sampling_x \
    --mcmc_n_step_x $mcmc_n_step_x \
    --mcmc_stepsize_x $mcmc_stepsize_x \
    --mcmc_noise_x $mcmc_noise_x \
    --mcmc_custom_stepsize \
    --mcmc_n_step_omi $mcmc_n_step_omi \
    --mcmc_stepsize_omi $mcmc_stepsize_omi \
    --mcmc_noise_omi $mcmc_noise_omi \
    --mcmc_normalize_omi \
    --proj_mode $proj_mode \
    --proj_noise_start $proj_noise_start \
    --proj_noise_end $proj_noise_end \
    --proj_const_omi $proj_const_omi \
    --proj_const $proj_const \
    --proj_dist $proj_dist \
    --num_layers $num_layers \
    --num_head $num_head \
    --eps $eps \
    --batch_size $batch_size \
    --nae_epochs $nae_epochs \
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
    --pretrained_ae $pretrained_ae \
    --pretrained_net_x $pretrained_net_x \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --gpus ${gpus[@]}


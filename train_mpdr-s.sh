#!/bin/bash

window_size=10
z_dim=256
temperature=1
gamma_vx=1
sampling_x=langevin
mcmc_n_step_x=5
mcmc_stepsize_x=10
mcmc_noise_x=0.005
mcmc_n_step_omi=2
mcmc_stepsize_omi=0.05
mcmc_noise_omi=0.02
proj_mode=uniform
proj_noise_start=0.05
proj_noise_end=0.3
proj_const_omi=0.0001
proj_const=0.0001
proj_dist=geodesic
num_layers=6
num_head=8
eps=1e-12
batch_size=128
nae_epochs=50
num_workers=32
lr=1e-4
dropout=0.1
accum_grad_batches=1
warmup_steps=1
scheduler_steps=49
weight_decay=0
beta1=0.9
beta2=0.999
save_dir="/raid/monsals/mpdr_physics"
name="mpdr-s-v1"
log_every_n_steps=10
save_top_k=1
pretrained_ae="/raid/monsals/mpdr_physics/checkpoints/ae-v1/last.ckpt"
init_net_x_ae="/raid/monsals/mpdr_physics/checkpoints/ae-v1/last.ckpt"
checkpoint_path="/raid/monsals/mpdr_physics/checkpoints"
checkpoint_name="mpdr-s-v1"
gpus=(4 5 6 7)

python -m train.mpdr-s \
    --latent_token \
    --pos_embedding \
    --window_size $window_size \
    --z_dim $z_dim \
    --temperature $temperature \
    --gamma_vx $gamma_vx \
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
    --use_recon_error \
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
    --init_net_x_ae $init_net_x_ae \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --gpus ${gpus[@]}


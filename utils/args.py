import argparse

'''
Parameters
'''
def ini_argparse():
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument("--bbh_dataset", type=str, default="/raid/monsals/gw/bbh_for_challenge.npy", help="BBH dataset path")
    parser.add_argument("--bkg_dataset", type=str, default="/raid/monsals/gw/background.npz", help="Background dataset path")
    parser.add_argument("--sglf_dataset", type=str, default="/raid/monsals/gw/sglf_for_challenge.npy", help="SGLF Dataset path")

    # AE arguments
    parser.add_argument("--in_features", type=int, default=2, help="input features")
    parser.add_argument("--seq_len", type=int, default=200, help="sequence length")
    parser.add_argument("--window_size", type=int, default=1, help="subsequence window size")
    parser.add_argument("--z_dim", type=int, default=32, help="latent space dimension")
    parser.add_argument("--num_layers", type=int, default=5, help="number of layers in AE encoder/decoder")
    parser.add_argument("--num_head", type=int, default=8, help="num heads in AE encoder/decoder (z_dim divisible by num_head)")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout ratio")
    parser.add_argument("--latent_token", action="store_true", default=None, help="represent the latent space with an extra token")
    parser.add_argument("--pos_embedding", action="store_true", default=None, help="set if learn positional embedding")
    parser.add_argument("--spherical", type=bool, default=True, help="Use spherical normalization")
    parser.add_argument("--out_spherical", type=bool, default=False, help="Output spherical normalization")
    parser.add_argument("--l2_norm_reg_enc", type=float, default=None, help="L2 norm regularization for encoder")
    parser.add_argument("--encoding_noise", type=float, default=None, help="Noise added during encoding")
    parser.add_argument("--reg_z_norm", type=float, default=None, help="Regularization for z normalization")
    parser.add_argument("--input_noise", type=float, default=None, help="Noise added to the input")
    parser.add_argument("--loss", type=str, default="l2", help="Loss function to use")
    parser.add_argument("--tau", type=float, default=0.1, help="Tau parameter")
    parser.add_argument("--perceptual_weight", type=float, default=None, help="Weight for perceptual loss")
    parser.add_argument("--condensation_weight", type=float, default=None, help="Weight for condensation loss")
    parser.add_argument("--learn_out_scale", type=bool, default=None, help="Learnable output scaling")

    # NAE
    parser.add_argument("--sampling_x", type=str, default="langevin", help="Sampling method for x")
    parser.add_argument("--mcmc_n_step_x", type=int, default=None, help="Number of MCMC steps for x")
    parser.add_argument("--mcmc_stepsize_x", type=float, default=None, help="Step size for MCMC on x")
    parser.add_argument("--mcmc_noise_x", type=float, default=None, help="Noise for MCMC on x")
    parser.add_argument("--mcmc_bound_x", type=tuple, default=None, help="Bounds for MCMC on x")
    parser.add_argument("--mcmc_n_step_omi", type=int, default=None, help="Number of MCMC steps for omi")
    parser.add_argument("--mcmc_stepsize_omi", type=float, default=None, help="Step size for MCMC on omi")
    parser.add_argument("--mcmc_noise_omi", type=float, default=None, help="Noise for MCMC on omi")
    parser.add_argument("--mcmc_stepsize_s", type=float, default=None, help="Step size for MCMC on s")
    parser.add_argument("--mcmc_noise_s", type=float, default=None, help="Noise for MCMC on s")
    parser.add_argument("--mcmc_custom_stepsize", action="store_true", default=False, help="Use custom step size for MCMC")
    parser.add_argument("--mcmc_normalize_omi", action="store_true", default=False, help="Normalize omi during MCMC")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--temperature_omi", type=float, default=1.0, help="Temperature for omi")
    parser.add_argument("--gamma_vx", type=float, default=None, help="Gamma value for vx")
    parser.add_argument("--gamma_neg_recon", type=float, default=None, help="Gamma value for negative reconstruction")
    parser.add_argument("--gamma_neg_b", type=float, default=None, help="Gamma value for negative b")
    parser.add_argument("--proj_mode", type=str, default="constant", help="Projection mode")
    parser.add_argument("--proj_noise_start", type=float, default=0.01, help="Starting noise for projection")
    parser.add_argument("--proj_noise_end", type=float, default=0.01, help="Ending noise for projection")
    parser.add_argument("--proj_const", type=float, default=1.0, help="Constant projection value")
    parser.add_argument("--proj_const_omi", type=float, default=None, help="Constant projection value for omi")
    parser.add_argument("--proj_dist", type=str, default="sum", help="Projection distance")
    parser.add_argument("--l2_norm_reg_netx", type=float, default=None, help="L2 norm regularization for net x")
    parser.add_argument("--energy_method", type=str, default=None, help="Energy method to use")
    parser.add_argument("--use_net_x", action="store_true", default=True, help="Use net x in the model")
    parser.add_argument("--use_recon_error", action="store_true", default=False, help="Use reconstruction error")
    parser.add_argument("--grad_clip_off", type=float, default=None, help="Disable gradient clipping")
    parser.add_argument("--recovery_gaussian_blur", type=float, default=None, help="Apply Gaussian blur during recovery")
    parser.add_argument("--apply_proj_grad", action="store_true", default=None, help="Apply projection gradients")
    parser.add_argument("--replay_ratio", type=float, default=None, help="Ratio of replay samples")
    parser.add_argument("--mh_omi", action="store_true", default=False, help="Use Metropolis-Hastings for omi")
    parser.add_argument("--mh_x", action="store_true", default=False, help="Use Metropolis-Hastings for x")
    parser.add_argument("--return_min", action="store_true", default=False, help="Return minimum values")
    parser.add_argument("--buffer_match_nearest", action="store_true", default=False, help="Match buffer to nearest values")
    parser.add_argument("--work_on_manifold", action="store_true", default=False, help="Work on the manifold")
    parser.add_argument("--conditional", action="store_true", default=False, help="Use conditional generation")
    parser.add_argument("--custom_netx_reg", action="store_true", default=False, help="Custom regularization for net x")
    parser.add_argument("--init_net_x_ae", type=str, default=None, help="Path of pretrained AE for init net_x")

    # Train arguments
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--ae_epochs", type=int, default=100, help="number of AE epochs")
    parser.add_argument("--nae_epochs", type=int, default=50, help="number of NAE epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=64, help="number of loader workers")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate of the optimiser")
    parser.add_argument("--eps", type=float, default=1e-8, help="value to prevent division by zero")
    parser.add_argument("-ag", "--accum_grad_batches", type=int, default=1, help="batches for gradient accumulation")
    parser.add_argument('-ws', '--warmup_steps', type=float, default=1, help='Maximum number of warmup steps')
    parser.add_argument("-st", "--scheduler_steps", type=float, default=126, help="scheduler steps in one cycle (iterations until the first restart)")
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4, help="weight_decay of the optimiser")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="AdamW first beta value")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="AdamW second beta value")
    parser.add_argument("--pretrained_ae", type=str, default=None, help="path of pretrained AE model")
    parser.add_argument("--save_dir", type=str, default="/raid/monsals/nae_physics", help="Log save directory")
    parser.add_argument("--name", type=str, default="v1", help="model name")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="steps between logs")
    parser.add_argument("--save_top_k", type=int, default=5, help="Save top k checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="/raid/monsals/nae_physics/checkpoints", help="Checkpoint path")
    parser.add_argument("--checkpoint_name", type=str, default="v1", help="Checkpoint name")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Name of the checkpoint to load")
    parser.add_argument('--gpus', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=[0],  # Default list
                        help='List of GPUs to use (more than 1 GPU will run the training in parallel)'
                        )

    return parser


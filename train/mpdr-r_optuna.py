import os
import sys
import pickle
import inspect
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import optuna

module_path = os.path.abspath('..')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from dataset import GWDataset, split_dataset, create_data_loader
from models import Encoder, Decoder, NAELightningModel
from models.mpdr import AE, MPDR_Single
from utils import ini_argparse

def retrieve_args(module, args):
    signature = inspect.signature(module)
    params = set(signature.parameters.keys())
    args = {key: value for key, value in vars(args).items() if key in params}
    return args

class NaNPruningCallback(Callback):
    def __init__(self, trial):
        self.trial = trial

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        train_loss = trainer.callback_metrics.get("train_loss")
        
        # Check if val_dice is NaN
        if train_loss is not None and (train_loss.isnan() or train_loss.isinf()):
            # Log that NaN was encountered and prune the trial
            self.trial.set_user_attr("nan_encountered", True)
            print("NaN encountered in validation metric. Pruning trial.")
            raise optuna.TrialPruned()

parser = ini_argparse()
args = parser.parse_args()

# Nicely print all arguments
print("\nArguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
    
nb_gpus = len(args.gpus)
gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])

# Manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# Datasets
bbh_dataset = GWDataset(args.bbh_dataset, args.augment)
bkg_dataset = GWDataset(args.bkg_dataset, args.augment)
sglf_dataset = GWDataset(args.sglf_dataset)

indist_dataset = ConcatDataset([bbh_dataset, bkg_dataset])
ood_dataset = sglf_dataset

# split datasets
indist_train_set, indist_val_set = split_dataset(indist_dataset, train_ratio=0.8)

# loaders
loader_args = retrieve_args(create_data_loader, args)
indist_train_loader = create_data_loader(indist_train_set, shuffle=True, **loader_args)
indist_val_loader = create_data_loader(indist_val_set, shuffle=False, **loader_args)
ood_val_loader = create_data_loader(ood_dataset, shuffle=False, **loader_args)

# Calculate arguments for scheduler
args.warmup_steps = int(len(indist_train_loader) * args.warmup_steps // (args.accum_grad_batches * nb_gpus))
args.scheduler_steps = int(len(indist_train_loader) * args.scheduler_steps // (args.accum_grad_batches * nb_gpus))

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    mcmc_n_step_x = trial.suggest_int("mcmc_n_step_x", 1, 20)
    mcmc_stepsize_x = trial.suggest_int("mcmc_stepsize_x", 1, 20)
    mcmc_noise_x = trial.suggest_float("mcmc_noise_x", 0.001, 0.01)
    mcmc_n_step_omi = trial.suggest_int("mcmc_n_step_omi", 0, 10)
    mcmc_stepsize_omi = trial.suggest_float("mcmc_stepsize_omi", 0.02, 1)
    mcmc_noise_omi = trial.suggest_float("mcmc_noise_omi", 0.0, 0.1)

    args.lr = lr
    args.mcmc_n_step_x = mcmc_n_step_x
    args.mcmc_stepsize_x = mcmc_stepsize_x
    args.mcmc_noise_x = mcmc_noise_x
    args.mcmc_n_step_omi = mcmc_n_step_omi
    args.mcmc_stepsize_omi = mcmc_stepsize_omi
    args.mcmc_noise_omi = mcmc_noise_omi

    print(f"args.lr = {args.lr}")
    print(f"args.mcmc_n_step_x = {args.mcmc_n_step_x}")
    print(f"args.mcmc_stepsize_x = {args.mcmc_stepsize_x}")
    print(f"args.mcmc_noise_x = {args.mcmc_noise_x}")
    print(f"args.mcmc_n_step_omi = {args.mcmc_n_step_omi}")
    print(f"args.mcmc_stepsize_omi = {args.mcmc_stepsize_omi}")
    print(f"args.mcmc_noise_omi = {args.mcmc_noise_omi}")

    enc_args = retrieve_args(Encoder, args)
    ae_args = retrieve_args(AE, args)
    nae_args = retrieve_args(MPDR_Single, args)

    encoder = Encoder(**enc_args)
    decoder = Decoder(**enc_args)
    ae_args2 = ae_args.copy()
    if 'learn_out_scale' in ae_args2:
        ae_args2['learn_out_scale'] = None
    ae = AE(encoder, decoder, **ae_args2)

    encoder = Encoder(**enc_args)
    decoder = Decoder(**enc_args)
    varnet_x = AE(encoder, decoder, **ae_args)

    nae = MPDR_Single(
        ae=ae, net_x=varnet_x, **nae_args,
        )
    nan_callback = NaNPruningCallback(trial)

    if args.pretrained_ae:
        state_dict = torch.load(args.pretrained_ae, map_location=torch.device('cpu'))['state_dict']

        # Modify the keys in the state dictionary
        updated_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.ae."):
                new_key = key.replace("model.ae.", "", 1)
                updated_state_dict[new_key] = value

        # Update the state dictionary with modified keys
        state_dict['model_state'] = updated_state_dict

        # Load the updated state dictionary into the model
        nae.ae.load_state_dict(state_dict['model_state'])
        print("Loaded pretrained AE model!")

    if args.pretrained_net_x:
        state_dict = torch.load(args.pretrained_net_x, map_location=torch.device('cpu'))['state_dict']

        # Modify the keys in the state dictionary
        updated_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.ae."):
                new_key = key.replace("model.ae.", "", 1)
                updated_state_dict[new_key] = value

        # Update the state dictionary with modified keys
        state_dict['model_state'] = updated_state_dict

        # Load the updated state dictionary into the model
        nae.net_x.load_state_dict(state_dict['model_state'])
        print("Loaded pretrained net_x model!")

    lightning_signature_args = retrieve_args(NAELightningModel, args)

    # Create lightning model
    lightning_model = NAELightningModel(
        model = nae,
        **lightning_signature_args,
    )

    # Create trainer module
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=6,
        callbacks=[nan_callback],
        accelerator="gpu",
        devices=nb_gpus,
        strategy="ddp" if nb_gpus > 1 else None,
        precision="32",
        deterministic=False,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    # Run the training
    trainer.fit(
        model=lightning_model,
        train_dataloaders=indist_train_loader,
        val_dataloaders=[indist_val_loader, ood_val_loader],
    )

    return trainer.callback_metrics["nae/auc_val"].item()

# Create an Optuna study and optimize it
study = optuna.load_study(study_name='distributed-example-mpdr-r', storage='sqlite:///example.db')
study.optimize(objective, n_trials=20)

# Display best trial
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
        print(f"    {key}: {value}")


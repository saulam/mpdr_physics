import os
import sys
import inspect
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
bbh_dataset = GWDataset(args.bbh_dataset)
bkg_dataset = GWDataset(args.bkg_dataset)
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

enc_args = retrieve_args(Encoder, args)
ae_args = retrieve_args(AE, args)
nae_args = retrieve_args(MPDR_Single, args)

encoder = Encoder(**enc_args)
decoder = Decoder(**enc_args)
ae = AE(encoder, decoder, learn_out_scale=False, **ae_args)

encoder = Encoder(**enc_args)
decoder = Decoder(**enc_args)
varnet_x = AE(encoder, decoder, **ae_args)

nae = MPDR_Single(
    ae=ae, net_x=varnet_x, **nae_args,
    )

encoder_params = sum(p.numel() for p in nae.ae.encoder.parameters())
print(f"Number of encoder parameters: {encoder_params}")
decoder_params = sum(p.numel() for p in nae.ae.decoder.parameters())
print(f"Number of decoder parameters: {decoder_params}")
varnet_x_params = sum(p.numel() for p in nae.net_x.parameters())
print(f"Number of varnet_x parameters: {varnet_x_params}")
total_params = sum(p.numel() for p in nae.parameters())
print(f"Total number of parameters: {total_params}")

if args.pretrained_ae:
    state_dict = torch.load(args.pretrained_ae, map_location=torch.device('cpu'))['state_dict']

    # Modify the keys in the state dictionary
    updated_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.ae"):
            new_key = key.replace("model.ae", "", 1)
            updated_state_dict[new_key] = value

    # Update the state dictionary with modified keys
    state_dict['model_state'] = updated_state_dict

    # Load the updated state dictionary into the model
    nae.ae.load_state_dict(state_dict['model_state'])
    print("Loaded pretrained AE model!")

if args.pretrained_net_x:
    state_dict = torch.load(args.init_net_x_ae, map_location=torch.device('cpu'))['state_dict']

    # Modify the keys in the state dictionary
    updated_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.net_x."):
            new_key = key.replace("model.net_x.", "", 1)
            updated_state_dict[new_key] = value

    # Update the state dictionary with modified keys
    state_dict['model_state'] = updated_state_dict

    # Load the updated state dictionary into the model
    nae.net_x.load_state_dict(state_dict['model_state'])
    print("Loaded pretrained net_x model!")

# Calculate arguments for scheduler
args.warmup_steps = int(len(indist_train_loader) * args.warmup_steps // args.accum_grad_batches)
args.scheduler_steps = int(len(indist_train_loader) * args.scheduler_steps // args.accum_grad_batches)

lightning_signature_args = retrieve_args(NAELightningModel, args)

# Create lightning model
lightning_model = NAELightningModel(
    model = nae,
    **lightning_signature_args,
)

# Define logger and checkpoint
logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name)
checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_path + "/" + args.checkpoint_name,
                                      save_last=True, save_top_k=args.save_top_k, mode='max', monitor="nae/auc_val")

# Log the hyperparameters
logger.log_hyperparams(vars(args))
tb_logger.log_hyperparams(vars(args))

# Create trainer module
trainer = pl.Trainer(
    max_epochs=args.nae_epochs,
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    devices=nb_gpus,
    strategy="ddp" if nb_gpus > 1 else None,
    precision="32",
    logger=[logger, tb_logger],
    log_every_n_steps=args.log_every_n_steps,
    val_check_interval=100,
    deterministic=False,
    accumulate_grad_batches=args.accum_grad_batches,
)

# Run the training
trainer.fit(
    model=lightning_model,
    train_dataloaders=indist_train_loader,
    val_dataloaders=[indist_val_loader, ood_val_loader],
    ckpt_path="/".join([args.checkpoint_path, args.checkpoint_name, args.load_checkpoint]) if args.load_checkpoint else None,
)


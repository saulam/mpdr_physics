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
from models.mpdr import AE, MPDR_Single
from models import Encoder, Decoder, AELightningModel
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
bbh_dataset = GWDataset(args.bbh_dataset, args.augment)
bkg_dataset = GWDataset(args.bkg_dataset, args.augment)
indist_dataset = ConcatDataset([bbh_dataset, bkg_dataset])

# split datasets
indist_train_set, indist_val_set = split_dataset(indist_dataset, train_ratio=0.8)

# loaders
loader_args = retrieve_args(create_data_loader, args)
indist_train_loader = create_data_loader(indist_train_set, shuffle=True, **loader_args)
indist_val_loader = create_data_loader(indist_val_set, shuffle=False, **loader_args)

enc_args = retrieve_args(Encoder, args)
ae_args = retrieve_args(AE, args)
nae_args = retrieve_args(MPDR_Single, args)

encoder = Encoder(**enc_args)
decoder = Decoder(**enc_args)

ae = AE(encoder, decoder, **ae_args)
varnet_x = Encoder(mlp_head=True, **enc_args)

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

# Calculate arguments for scheduler
args.warmup_steps = int(len(indist_train_loader) * args.warmup_steps // (args.accum_grad_batches * nb_gpus))
args.scheduler_steps = int(len(indist_train_loader) * args.scheduler_steps // (args.accum_grad_batches * nb_gpus))

lightning_signature_args = retrieve_args(AELightningModel, args)

# Create lightning model
lightning_model = AELightningModel(
    model = nae,
    **lightning_signature_args,
)

# Define logger and checkpoint
logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name)
checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_path + "/" + args.checkpoint_name,
                                      save_last=True, save_top_k=args.save_top_k, monitor="ae/val_loss")

# Log the hyperparameters
logger.log_hyperparams(vars(args))
tb_logger.log_hyperparams(vars(args))

# Create trainer module
trainer = pl.Trainer(
    max_epochs=args.ae_epochs,
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    devices=nb_gpus,
    strategy="ddp" if nb_gpus > 1 else None,
    precision="32",
    logger=[logger, tb_logger],
    log_every_n_steps=args.log_every_n_steps,
    deterministic=False,
    accumulate_grad_batches=args.accum_grad_batches,
)

# Run the training
trainer.fit(
    model=lightning_model,
    train_dataloaders=indist_train_loader,
    val_dataloaders=indist_val_loader,
    ckpt_path="/".join([args.checkpoint_path, args.checkpoint_name, args.load_checkpoint]) if args.load_checkpoint else None,
)


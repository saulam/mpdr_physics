import torch
import pytorch_lightning as pl
from utils import roc_btw_arr, CustomLambdaLR, CombinedScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class DivergenceDetector():
    """
    detects divergence in the training process from observing negative sample energy.
    Collect negative sample energy and compute Gaussian statistics by rolling mean and rolling standard deviation.
    Declare divergence when negative sample energy is larger than 6-sigma interval from the mean.
    """
    def __init__(self, window_size=1000, disable=False):
        self.window_size = window_size
        self.neg_sample_energy = []
        self.mean = 0
        self.std = 0
        self.disable = disable

    def update(self, neg_sample_energy):
        self.neg_sample_energy.append(neg_sample_energy)
        if len(self.neg_sample_energy) > self.window_size:
            self.neg_sample_energy.pop(0)
        self.mean = np.mean(self.neg_sample_energy)
        self.std = np.std(self.neg_sample_energy)

    def detect(self):
        if self.disable:
            return False
        if len(self.neg_sample_energy) < self.window_size:
            return False
        if self.neg_sample_energy[-1] > self.mean + 6 * max(self.std, 0.1) and self.neg_sample_energy[-1] > 0.2:
            return True
        return False


class AELightningModel(pl.LightningModule):
    def __init__(self, model, warmup_steps=1, scheduler_steps=100, lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=1e-4, eps=1e-8):
        super().__init__()

        self.model = model
        self.warmup_steps = warmup_steps
        self.cosine_annealing_steps = scheduler_steps
        self.lr = lr
        self.betas = (beta1, beta2)
        self.weight_decay = weight_decay
        self.eps = eps
        self.automatic_optimization = False


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups


    def on_train_epoch_start(self):
        """Hook to be called at the start of each training epoch."""
        train_loader = self.trainer.train_dataloader
        train_loader.dataset.datasets.dataset.set_training_mode(True)


    def on_validation_epoch_start(self):
        """Hook to be called at the start of each validation epoch."""
        val_loader = self.trainer.val_dataloaders[0]
        val_loader.dataset.dataset.set_training_mode(False)


    def training_step(self, batch, batch_idx):
        """Training step for a batch of data."""
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        outputs = self.model.train_step_ae(
            x = batch,
            optimizer = optimizer,
            manual_backward = self.manual_backward,
            scheduler = scheduler,
            clip_grad = None,
        )
        batch_size = batch.shape[0]
        lr = self.optimizers().param_groups[0]['lr']

        for key, val in outputs.items():
            if key.endswith('_'):
                self.log(f"ae/" + key, val, batch_size=batch_size, prog_bar=False, sync_dist=True)
            elif key=="loss":
                self.log(f"ae/train_loss", val, batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log(f"ae/lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)
 
        return


    def validation_step(self, batch, batch_idx):
        """Validation step for a batch of data."""
        outputs = self.model.validation_step_ae(batch)
        batch_size = batch.shape[0]

        # Log losses
        for key, val in outputs.items():
            if key.endswith('_'):
                self.log(f"ae/" + key, val, batch_size=batch_size, prog_bar=False, sync_dist=True)
            elif key=="loss":
                self.log(f"ae/val_loss", val, batch_size=batch_size, prog_bar=True, sync_dist=True)

        return


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        # Optimiser
        optimizer = torch.optim.AdamW(
            list(self.model.ae.parameters()),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )

        if self.warmup_steps==0 and self.cosine_annealing_steps==0:
            return optimizer
        else:
            if self.warmup_steps == 0:
                warmup_scheduler = None
            else:
                # Warm-up scheduler
                warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)
            
            if self.cosine_annealing_steps == 0:
                cosine_scheduler = None
            else:
                # Cosine annealing scheduler
                cosine_scheduler = CosineAnnealingWarmRestarts(
                    optimizer=optimizer,
                    T_0=self.cosine_annealing_steps,
                    eta_min=0
                )

            # Combine both schedulers
            combined_scheduler = CombinedScheduler(
                optimizer=optimizer,
                scheduler1=warmup_scheduler,
                scheduler2=cosine_scheduler,
                warmup_steps=self.warmup_steps,
            )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


class NAELightningModel(pl.LightningModule):
    def __init__(self, model, warmup_steps=1, scheduler_steps=100, lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=1e-4, eps=1e-8):
        super().__init__()

        self.model = model
        self.warmup_steps = warmup_steps
        self.cosine_annealing_steps = scheduler_steps
        self.lr = lr
        self.betas = (beta1, beta2)
        self.weight_decay = weight_decay
        self.eps = eps
        self.automatic_optimization = False


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups


    def on_train_epoch_start(self):
        """Hook to be called at the start of each training epoch."""
        train_loader = self.trainer.train_dataloader
        train_loader.dataset.datasets.dataset.set_training_mode(True)


    def on_validation_epoch_start(self):
        """Hook to be called at the start of each validation epoch."""
        val_loader = self.trainer.val_dataloaders[0]
        val_loader.dataset.dataset.set_training_mode(False)


    def training_step(self, batch, batch_idx):
        """Training step for a batch of data."""
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        outputs = self.model.train_step(
            x = batch,
            optimizer = optimizer,
            manual_backward = self.manual_backward,
            scheduler = scheduler,
            clip_grad = None,
        )
        batch_size = batch.shape[0]
        lr = self.optimizers().param_groups[0]['lr']

        for key, val in outputs.items():
            if key.endswith('_'):
                self.log(f"nae/" + key, val, batch_size=batch_size, prog_bar=False, sync_dist=True)
            elif key=="loss":
                self.log(f"nae/train_loss", val, batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log(f"nae/lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)
 
        return


    def predict(self, dl):
        """run prediction for the whole dataset"""
        l_result = []
        for x in dl:
            with torch.no_grad():
                pred = self.model.predict(x.to(self.device)).detach().cpu()
            l_result.append(pred)
        return torch.cat(l_result)


    def validation_step(self, *args, **kwargs):
        """Validation step for a batch of data.""" 
        return


    def validation_epoch_end(self, outputs):
        indist_val_loader = self.trainer.val_dataloaders[0]
        ood1_val_loader = self.trainer.val_dataloaders[1]
        ood2_val_loader = self.trainer.val_dataloaders[2]

        in_pred = self.predict(indist_val_loader)
        ood1_pred = self.predict(ood1_val_loader)
        ood2_pred = self.predict(ood2_val_loader)
        auc_val1 = roc_btw_arr(ood1_pred, in_pred)
        auc_val2 = roc_btw_arr(ood2_pred, in_pred)
        auc_val_avg = (auc_val1 + auc_val2) / 2.

        # Log losses
        self.log(f"nae/auc_val_avg", auc_val_avg, prog_bar=True, sync_dist=True)
        self.log(f"nae/auc_val1", auc_val1, prog_bar=False, sync_dist=True)
        self.log(f"nae/auc_val2", auc_val2, prog_bar=False, sync_dist=True)

        indist_val_loader.dataset.dataset.set_training_mode(True)

        return

    
    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        # Optimiser
        optimizer = torch.optim.AdamW(
            list(self.model.net_x.parameters()),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )

        if self.warmup_steps==0 and self.cosine_annealing_steps==0:
            return optimizer
        else:
            if self.warmup_steps == 0:
                warmup_scheduler = None
            else:
                # Warm-up scheduler
                warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)

            if self.cosine_annealing_steps == 0:
                cosine_scheduler = None
            else:
                # Cosine annealing scheduler
                cosine_scheduler = CosineAnnealingWarmRestarts(
                    optimizer=optimizer,
                    T_0=self.cosine_annealing_steps,
                    eta_min=0
                )

            # Combine both schedulers
            combined_scheduler = CombinedScheduler(
                optimizer=optimizer,
                scheduler1=warmup_scheduler,
                scheduler2=cosine_scheduler,
                warmup_steps=self.warmup_steps,
            )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


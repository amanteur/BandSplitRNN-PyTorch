import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer, lr_scheduler
from typing import Tuple, Dict
from omegaconf import DictConfig


class PLModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            featurizer: nn.Module,
            inverse_featurizer: nn.Module,
            augmentations: nn.Module,
            opt: Optimizer,
            sch: lr_scheduler._LRScheduler,
            hparams: DictConfig
    ):
        super().__init__()

        # augmentations
        self.augmentations = augmentations

        # featurizers
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer

        # model
        self.model = model

        # losses
        self.mae_specR = nn.L1Loss()
        self.mae_specI = nn.L1Loss()
        self.mae_time = nn.L1Loss()

        # opts
        self.opt = opt
        self.sch = sch

        # logging
        self.save_hyperparameters(hparams)

    def on_after_batch_transfer(
            self, batch, dataloader_idx
    ) -> torch.Tensor:
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        Output shape: [batch_size, n_sources, n_channels, freq, time]
        """
        batch = self.augmentations(batch)
        batch = self.featurizer(batch)
        return batch

    def training_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        """
        Input shape: [batch_size, n_sources, n_channels, freq, time]
        Output: loss
        """
        loss_dict, loss = self.step(batch)

        # logging
        for k in loss_dict:
            self.log(f"train/{k}", loss_dict[k])
        self.log("train/loss", loss)

        return loss

    def validation_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        loss_dict, loss = self.step(batch)
        # logging
        for k in loss_dict:
            self.log(f"val/{k}", loss_dict[k])
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def step(
            self, batch
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        mix, tgt = batch[:, 0], batch[:, 1]
        tgt_pred = self.model(mix)
        loss_dict = self.compute_losses(tgt_pred, tgt)
        loss = sum(loss_dict.values())

        return loss_dict, loss

    def compute_losses(
            self,
            tgt_pred: torch.Tensor,
            tgt_real: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # frequency domain
        lossR = self.mae_specR(
            tgt_pred.real, tgt_real.real
        )
        lossI = self.mae_specI(
            tgt_pred.imag, tgt_real.imag
        )

        # time domain
        lossT = self.mae_time(
            self.inverse_featurizer(tgt_pred),
            self.inverse_featurizer(tgt_real)
        )

        return {
            "lossSpecR": lossR,
            "lossSpecI": lossI,
            "lossTime": lossT
        }

    def configure_optimizers(self):
        return [self.opt], [self.sch]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


class GradNormCallback(pl.Callback):
    """
    Logs the gradient norm.
    """
    def on_after_backward(self, trainer, model):
        model.log("my_model/grad_norm", self.gradient_norm(model))

    @staticmethod
    def gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

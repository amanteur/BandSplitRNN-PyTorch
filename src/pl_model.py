import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer, lr_scheduler
from typing import Dict
from omegaconf import DictConfig


class PLModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            featurizer: nn.Module,
            inverse_featurizer: nn.Module,
            opt: Optimizer,
            sch: lr_scheduler._LRScheduler,
            hparams: DictConfig
    ):
        super().__init__()

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
    ) -> Dict[str, torch.Tensor]:
        for k in batch:
            batch[k] = self.featurizer(batch[k])
        return batch

    def training_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        mix, tgt = batch['mix'], batch['tgt']
        mix = self.model(mix)
        loss_dict = self.compute_losses(mix, tgt)
        loss = sum(loss_dict.values())

        # logging
        for k in loss_dict:
            self.log(f"train/{k}", loss_dict[k])
        self.log("train/loss", loss)

        return loss

    def validation_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        mix, tgt = batch['mix'], batch['tgt']
        mix = self.model(mix)
        loss_dict = self.compute_losses(mix, tgt)
        loss = sum(loss_dict.values())

        # logging
        for k in loss_dict:
            self.log(f"val/{k}", loss_dict[k])
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def compute_losses(self, mix, tgt):
        # frequency domain
        lossR = self.mae_specR(mix.real, tgt.real)
        lossI = self.mae_specI(mix.imag, tgt.imag)

        # time domain
        mix = self.inverse_featurizer(mix)
        tgt = self.inverse_featurizer(tgt)
        lossT = self.mae_time(mix, tgt)

        return {
            "lossSpecR": lossR,
            "lossSpecI": lossI,
            "lossTime": lossT
        }

    def configure_optimizers(self):
        return [self.opt], [self.sch]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

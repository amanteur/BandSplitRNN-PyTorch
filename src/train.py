import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data import SourceSeparationDataset, collate_fn
from model import BandSplitRNN
from pl_model import PLModel

from typing import Tuple
from torch.optim import Optimizer, lr_scheduler


def initialize_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    train_dataset = SourceSeparationDataset(
        **cfg.train_dataset,
    )
    val_dataset = SourceSeparationDataset(
        **cfg.val_dataset,
    )
    train_loader = DataLoader(
        train_dataset,
        **cfg.train_loader,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        **cfg.val_loader,
        collate_fn=collate_fn
    )
    return (
        train_loader,
        val_loader
    )


def initialize_featurizer(
        cfg: DictConfig
) -> Tuple[nn.Module, nn.Module]:
    """
    Initializes direct and inverse featurizers for audio.
    """
    featurizer = instantiate(cfg.featurizer.direct_transform)
    inv_featurizer = instantiate(cfg.featurizer.inverse_transform)
    return featurizer, inv_featurizer


def initialize_model(
        cfg: DictConfig
) -> Tuple[nn.Module, Optimizer, lr_scheduler._LRScheduler]:
    """
    Initializes model from configuration file.
    """
    model = BandSplitRNN(
        **cfg.model
    )
    if cfg.ckpt_path is not None:
        state_dict = torch.load(cfg.ckpt_path)
        _ = model.load_state_dict(state_dict, strict=True)

    opt = instantiate(
        cfg.opt,
        params=model.parameters()
    )
    sch = instantiate(
        cfg.sch,
        optimizer=opt
    )
    return model, opt, sch


def initialize_utils(
        cfg: DictConfig
):
    # change model and logs saving directory to logging directory of hydra
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_dir = hydra_cfg['runtime']['output_dir']
    cfg.logger.save_dir = save_dir + cfg.logger.save_dir
    cfg.callbacks.model_ckpt.dirpath = save_dir + cfg.callbacks.model_ckpt.dirpath
    # initialize logger and callbacks
    logger = instantiate(cfg.logger)
    callbacks = list(instantiate(cfg.callbacks).values())
    return logger, callbacks


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # initializing loaders, featurizers
    train_loader, val_loader = initialize_loaders(cfg)
    featurizer, inverse_featurizer = initialize_featurizer(cfg)
    # TODO: augmentations = initialize_augmentations(cfg)

    # initializing model, optimizer, and scheduler
    model, opt, sch = initialize_model(cfg)

    # initialize other stuff
    logger, callbacks = initialize_utils(cfg)

    # initialize lightning model and trainer
    plmodel = PLModel(
        model,
        featurizer, inverse_featurizer,
        opt, sch,
        cfg
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    # fit model
    trainer.fit(
        plmodel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    my_app()

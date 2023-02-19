import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SourceSeparationDataset, collate_fn
from model import BandSplitRNN

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
) -> [nn.Module, nn.Module]:
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



@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train_loader, val_loader = initialize_loaders(cfg)
    featurizer = initialize_featurizer(cfg)
    model, opt, sch = initialize_model(cfg)


if __name__ == "__main__":
    my_app()
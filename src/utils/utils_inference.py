import torch
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
from typing import Tuple


def load_pl_state_dict(
        path: Path
) -> OrderedDict[str, torch.Tensor]:
    """
    Loads and preprocesses pytorch-lightning state dict
    """
    sd = torch.load(path)
    new_sd = OrderedDict()
    for k, v in sd['state_dict'].items():
        if 'model' in k:
            new_sd[k.replace('model.', '')] = v
    return new_sd


def get_minibatch(
        n_fragments: int, batch_size: int = 32
) -> Tuple[int, int]:
    """
    Yields indices of the start and the end of minibatch of size 'batch_size'
    """
    starts = torch.arange(0, n_fragments, batch_size)
    for s in tqdm(starts):
        yield s, s + batch_size


def overlap_add(
        y: torch.Tensor, y_tgt: torch.Tensor, step: int
) -> torch.Tensor:
    """
    overlap-add algorithm
    """
    y_overlapped = torch.zeros_like(y_tgt)
    start = 0
    end = y.shape[-1]
    for y_chunk in y:
        y_overlapped[:, start:end] += y_chunk
        start += step
        end += step
    return y_overlapped
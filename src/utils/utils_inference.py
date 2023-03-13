import torch
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
from typing import Tuple


def load_pl_state_dict(
        path: Path, device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """
    Loads and preprocesses pytorch-lightning state dict
    """
    sd = torch.load(path, map_location=device)
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
        y_in: torch.Tensor, duration: int, hl: int
) -> torch.Tensor:
    """
    overlap-add algorithm
    """
    wl = y_in.shape[-1]
    start = 0

    y_out = torch.zeros((y_in.shape[1], duration))
    for y_i in y_in:
        y_out[:, start:start + wl] += y_i
        start += hl
    return y_out / (wl / hl)

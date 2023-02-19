import torch
from typing import List, Dict, Tuple


def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collates list of mix and target torch.Tensors to torch.Tensor.
    Channels are collapsed to batch dimension.
    Tensor's Input shape: Batch of [1, n_timestamps]
    Tensor's Output shape: [batch_size, 1, n_timestamps]
    """
    batch_mix = []
    batch_tgt = []

    for mix, tgt in batch:
        batch_mix.append(mix)
        batch_tgt.append(tgt)

    return {
        "mix": torch.cat(batch_mix, dim=0).unsqueeze(1),
        "tgt": torch.cat(batch_tgt, dim=0).unsqueeze(1)
    }

import torch
from tqdm import tqdm


def get_minibatch(n_fragments: int, batch_size: int = 32):
    starts = torch.arange(0, n_fragments, batch_size)
    for s in tqdm(starts):
        yield s, s + batch_size


def overlap_add(y: torch.Tensor, y_tgt: int, step: int):
    y_overlapped = torch.zeros_like(y_tgt)
    start = 0
    end = y.shape[-1]
    for y_chunk in y:
        y_overlapped[:, start:end] += y_chunk
        start += step
        end += step
    return y_overlapped

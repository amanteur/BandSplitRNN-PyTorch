import numpy as np
import torch
from museval.metrics import bss_eval
from typing import Tuple


def compute_uSDR(
        y_hat: np.ndarray,
        y_tgt: np.ndarray,
        eps: float = 1e-7,
) -> float:
    """
    Computes SDR metric as in https://arxiv.org/pdf/2108.13559.pdf
    """
    return np.sum(y_tgt ** 2) / ((np.sum((y_tgt - y_hat) ** 2)) + eps)


def compute_SDRs(
        y_hat: torch.Tensor, y_tgt: torch.Tensor
) -> Tuple[float, float]:
    """
    Computes cSDR and uSDR as defined in paper
    """
    y_hat = y_hat.T.unsqueeze(0).numpy()
    y_tgt = y_tgt.T.unsqueeze(0).numpy()
    # bss_eval way
    cSDR, *_ = bss_eval(
        y_hat,
        y_tgt
    )
    cSDR = np.median(cSDR[~np.isnan(cSDR)])

    # as in music demixing challenge
    uSDR = compute_uSDR(
        y_hat,
        y_tgt
    )
    return cSDR, uSDR

import argparse
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from train import initialize_model, initialize_featurizer
from data import TestSourceSeparationDataset
from utils.utils_inference import get_minibatch, overlap_add
from utils.utils_test import compute_SDRs

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d',
    '--run-dir',
    type=str,
    required=True,
    help="Path to directory checkpoints, configs, etc"
)
parser.add_argument(
    '-s',
    '--eval-step',
    type=float,
    required=False,
    default=0.5,
    help="Step between audio chunks",
)


def initialize_logger(run_dir: str):
    return None


def run_test_loop(
        model: nn.Module, dataset: Dataset,
        featurizer: nn.Module, inverse_featurizer: nn.Module
):
    cSDRs = []
    uSDRs = []

    for b in dataset:
        y_mix, y_tgt, dur = b[0], b[1], b[2]
        input_length = y_mix.shape[-1]

        y_mix = featurizer(y_mix)
        y_tgt_hat = torch.empty_like(y_mix)

        for s, e in get_minibatch(y_mix.shape[0]):
            y_tgt_hat[s:e] = model(y_mix[s:e])
        y_tgt_hat = inverse_featurizer(y_tgt_hat, length=input_length)
        y_tgt_hat = overlap_add(y_tgt_hat, y_tgt, step=dataset.eval_step)
        cSDR, uSDR = compute_SDRs(y_tgt_hat, y_tgt)

        cSDRs.append(cSDR)
        uSDRs.append(uSDRs)

    return cSDRs, uSDRs


def run_test(
        ckpt_dir_path: Path,
        model: nn.Module,
        dataset: Dataset,
        featurizer: nn.Module, inverse_featurizer: nn.Module
):
    for ckpt_path in ckpt_dir_path.glob("*.ckpt"):
        state_dict = torch.load(ckpt_path)
        _ = model.load_state_dict(state_dict, strict=True)
        metrics = run_test_loop(
            model, dataset,
            featurizer, inverse_featurizer
        )
        print(metrics)
    return None


def main(
        run_dir: str,
        eval_step: float
):
    run_dir = Path(run_dir)
    cfg_path = run_dir / '.hydra/config.yaml'
    ckpt_dir_path = run_dir / 'weights'
    cfg = OmegaConf.load(cfg_path)

    dataset = TestSourceSeparationDataset(**cfg.test_dataset)
    dataset.eval_step = int(eval_step * cfg.test_dataset.sr)

    featurizer, inverse_featurizer = initialize_featurizer(cfg)
    model, *_ = initialize_model(cfg)

    run_test(
        ckpt_dir_path,
        model, dataset,
        featurizer, inverse_featurizer,
    )


if __name__ == '__main__':
    args = parser.parse_args()

    main(
        args.run_dir,
        args.eval_step
    )

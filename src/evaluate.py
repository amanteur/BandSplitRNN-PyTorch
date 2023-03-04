import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import TestSourceSeparationDataset
from train import initialize_model, initialize_featurizer
from utils.utils_inference import load_pl_state_dict, get_minibatch, overlap_add
from utils.utils_test import compute_SDRs

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d',
    '--run-dir',
    type=str,
    required=True,
    help="Path to directory checkpoints, configs, etc"
)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename=f'{args.run_dir}/test.log',
    filemode='w'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ = torch.set_grad_enabled(False)


def run_eval_loop(
        model: nn.Module, dataset: Dataset,
        featurizer: nn.Module, inverse_featurizer: nn.Module
):
    metrics = defaultdict(list)

    for b in dataset:
        y_mix, y_tgt, dur = b[0], b[1], b[2]
        input_length = y_mix.shape[-1]

        y_mix = y_mix.to(device)
        y_mix = featurizer(y_mix)
        y_tgt_hat = torch.empty_like(y_mix)

        for s, e in get_minibatch(y_mix.shape[0]):
            y_tgt_hat[s:e] = model(y_mix[s:e])
        y_tgt_hat = inverse_featurizer(y_tgt_hat, length=input_length)
        y_tgt_hat = y_tgt_hat.cpu()
        y_tgt_hat = overlap_add(y_tgt_hat, y_tgt, step=dataset.hop_size)
        cSDR, uSDR = compute_SDRs(y_tgt_hat, y_tgt)

        metrics['cSDR'].append(cSDR)
        metrics['uSDR'].append(uSDR)

    metrics['cSDR'] = np.array(metrics['cSDR'])
    metrics['uSDR'] = np.array(metrics['uSDR'])

    return metrics


def run_eval(
        ckpt_dir_path: Path,
        model: nn.Module,
        dataset: Dataset,
        featurizer: nn.Module, inverse_featurizer: nn.Module
):
    for ckpt_path in ckpt_dir_path.glob("*.ckpt"):
        logger.info(f"Evaluating checkpoint - {ckpt_path.name}")
        state_dict = load_pl_state_dict(ckpt_path)
        _ = model.load_state_dict(state_dict, strict=True)

        metrics = run_eval_loop(
            model, dataset,
            featurizer, inverse_featurizer
        )
        for m in metrics:
            logger.info(
                f"Metric - {m}, mean - {metrics[m].mean():.3f}, std - {metrics[m].std():.3f}"
            )

    return None


def main(run_dir: str):
    logger.info("Starting evaluation...")
    run_dir = Path(run_dir)
    cfg_path = run_dir / 'tb_logs/hparams.yaml'
    ckpt_dir_path = run_dir / 'weights'
    cfg = OmegaConf.load(cfg_path)
    logger.info(f"Used model: {cfg_path}")

    logger.info("Initializing modules...")
    dataset = TestSourceSeparationDataset(**cfg.test_dataset)
    featurizer, inverse_featurizer = initialize_featurizer(cfg)
    model, *_ = initialize_model(cfg)
    _ = model.eval()

    _ = featurizer.to(device)
    _ = inverse_featurizer.to(device)
    _ = model.to(device)

    logger.info("Starting evaluation run...")
    run_eval(
        ckpt_dir_path,
        model, dataset,
        featurizer, inverse_featurizer,
    )


if __name__ == '__main__':
    main(args.run_dir)

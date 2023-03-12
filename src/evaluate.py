import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import TestSourceSeparationDataset
from train import initialize_model, initialize_featurizer
from utils.utils_inference import load_pl_state_dict, get_minibatch, overlap_add
from utils.utils_test import compute_SDRs


class EvaluateProgram:

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG_PATH = '{}/tb_logs/hparams.yaml'
    CKPT_DIR = '{}/weights'

    def __init__(
            self,
            run_dir: str
    ):
        self.cfg_path = Path(self.CFG_PATH.format(run_dir))
        self.ckpt_dir = Path(self.CKPT_DIR.format(run_dir))

        self.cfg = OmegaConf.load(self.cfg_path)
        logger.info(f"Used model: {self.cfg_path}")

        logger.info("Initializing modules...")
        model, featurizer, inverse_featurizer, dataset = self.initialize_all()
        self.model = model
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer
        self.dataset = dataset

    def initialize_all(self):
        featurizer, inverse_featurizer = initialize_featurizer(self.cfg)
        model, *_ = initialize_model(self.cfg)
        _ = model.eval()

        _ = featurizer.to(self.DEVICE)
        _ = inverse_featurizer.to(self.DEVICE)
        _ = model.to(self.DEVICE)

        dataset = TestSourceSeparationDataset(**self.cfg.test_dataset)

        return model, featurizer, inverse_featurizer, dataset

    @torch.no_grad()
    def inference_one(
            self,
            y: torch.Tensor,
            dur: int,
    ) -> torch.Tensor:
        # to device
        yT = y.to(self.DEVICE)
        # run through featurizer
        yS = self.featurizer(yT)
        # run through model
        yS_hat = torch.empty_like(yS)
        for s, e in get_minibatch(yS.shape[0]):
            yS_hat[s:e] = self.model(yS[s:e])
        # run through inverse featurizer
        yT_hat = self.inverse_featurizer(yS_hat, length=yT.shape[-1]).cpu()
        # overlap-add moment
        yT_hat = overlap_add(yT_hat, dur, hl=self.dataset.hop_size)
        # delete padded chunks
        yT_hat = yT_hat[:, self.dataset.pad_size:-self.dataset.pad_size]
        return yT_hat

    @torch.no_grad()
    def evaluate_all(self) -> Dict[str, np.ndarray]:
        metrics = defaultdict(list)
        for item in self.dataset:
            # load dataset item
            y, y_dur, y_tgt = item
            # run inference on mixture
            y_hat = self.inference_one(y, y_dur)
            # compute and save metrics
            cSDR, uSDR = compute_SDRs(y_hat, y_tgt)

            metrics['cSDR'].append(cSDR)
            metrics['uSDR'].append(uSDR)

        metrics['cSDR'] = np.array(metrics['cSDR'])
        metrics['uSDR'] = np.array(metrics['uSDR'])
        return metrics

    def evaluate(self) -> None:
        for ckpt_path in self.ckpt_dir.glob("*.ckpt"):
            logger.info(f"Evaluating checkpoint - {ckpt_path.name}")
            state_dict = load_pl_state_dict(ckpt_path, device=self.DEVICE)
            _ = self.model.load_state_dict(state_dict, strict=True)
            metrics = self.evaluate_all()
            for m in metrics:
                logger.info(
                    f"Metric - {m}, mean - {metrics[m].mean():.3f}, std - {metrics[m].std():.3f}"
                )
        return None


def main(run_dir: str):
    logger.info("Starting evaluation...")
    program = EvaluateProgram(run_dir)
    logger.info("Starting evaluation run...")
    program.evaluate()


if __name__ == '__main__':
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

    main(args.run_dir)

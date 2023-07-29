import argparse
import logging
import typing as tp
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from separator import Separator
from data import EvalSourceSeparationDataset
from utils.utils_inference import load_pl_state_dict
from utils.utils_test import compute_SDRs


class EvaluateProgram:
    CFG_PATH = '{}/tb_logs/hparams.yaml'
    CKPT_DIR = '{}/weights'

    def __init__(
            self,
            run_dir: str,
            device: str = 'cuda'
    ):
        # paths
        self.cfg_path = Path(self.CFG_PATH.format(run_dir))
        self.ckpt_dir = Path(self.CKPT_DIR.format(run_dir))

        # config params
        self.cfg = OmegaConf.load(self.cfg_path)
        logger.info(f"Used model: {self.cfg_path}")

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        )

        logger.info("Initializing the dataset...")
        self.dataset = EvalSourceSeparationDataset(mode='test', **self.cfg.test_dataset)
        logger.info("Initializing the separator...")
        self.cfg['audio_params'] = self.cfg.test_dataset
        self.sep = Separator(self.cfg, None)
        _ = self.sep.eval()
        _ = self.sep.to(self.device)

    def run_one_ckpt(self) -> tp.Dict[str, np.ndarray]:
        metrics = defaultdict(list)
        for y, y_tgt in self.dataset:
            # send to device
            y = y.to(self.device)

            # run inference on mixture
            y_hat = self.sep(y).cpu()

            # compute and save metrics
            cSDR, uSDR = compute_SDRs(y_hat, y_tgt)

            metrics['cSDR'].append(cSDR)
            metrics['uSDR'].append(uSDR)

        metrics['cSDR'] = np.array(metrics['cSDR'])
        metrics['uSDR'] = np.array(metrics['uSDR'])
        return metrics

    def run(self) -> None:
        # iterate over checkpoints
        for ckpt_path in self.ckpt_dir.glob("*.ckpt"):
            logger.info(f"Evaluating checkpoint - {ckpt_path.name}")
            state_dict = load_pl_state_dict(ckpt_path, device=self.device)
            _ = self.sep.model[1].load_state_dict(state_dict, strict=True)
            metrics = self.run_one_ckpt()
            for m in metrics:
                logger.info(
                    f"Metric - {m}, mean - {metrics[m].mean():.3f}, std - {metrics[m].std():.3f}"
                )
        return None


def main(args):
    logger.info("Starting evaluation...")
    args = vars(args)
    program = EvaluateProgram(**args)
    logger.info("Starting evaluation run...")
    program.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--run-dir',
        type=str,
        required=True,
        help="Path to directory checkpoints, configs, etc"
    )
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cuda',
        help="Device name - either 'cuda', or 'cpu'."
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

    main(args)

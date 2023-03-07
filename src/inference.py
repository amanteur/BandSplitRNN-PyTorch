import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import soundfile as sf
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import InferenceSourceSeparationDataset
from train import initialize_model, initialize_featurizer
from utils.utils_inference import load_pl_state_dict, get_minibatch, overlap_add

SAVED_MODELS_DIR = Path("./saved_models")

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--in-path',
    type=str,
    required=True,
    help="Path to the input directory/file with .wav/.mp3 extensions."
)
parser.add_argument(
    '-o',
    '--out-path',
    type=str,
    required=True,
    help="Path to the output directory. Files will be saved in .wav format with sr=44100."
)
parser.add_argument(
    '-t',
    '--target',
    type=str,
    required=False,
    default='vocals',
    help="Name of the target source to extract. "
)
parser.add_argument(
    '-c',
    '--ckpt-path',
    type=str,
    required=False,
    default=None,
    help="Path to model's checkpoint. If not specified, the .ckpt from SAVED_MODELS_DIR/{target} is used."
)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ = torch.set_grad_enabled(False)


def initialize_all(
        target: str,
        ckpt_path: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module, DictConfig]:
    tgt_dir = SAVED_MODELS_DIR / target

    # load cfg
    cfg_path = tgt_dir / 'hparams.yaml'
    if ckpt_path is None:
        ckpt_path = next(iter(tgt_dir.glob('*.ckpt')))

    cfg = OmegaConf.load(cfg_path)

    # load featurizer, inverse_featurizer, and model
    featurizer, inverse_featurizer = initialize_featurizer(cfg)
    model, *_ = initialize_model(cfg)

    # load checkpoint
    state_dict = load_pl_state_dict(ckpt_path, device=device)
    _ = model.load_state_dict(state_dict, strict=True)
    _ = model.eval()

    # send all to device
    _ = featurizer.to(device)
    _ = inverse_featurizer.to(device)
    _ = model.to(device)

    return model, featurizer, inverse_featurizer, cfg


@torch.no_grad()
def run_inference(
        model: nn.Module,
        featurizer: nn.Module,
        inverse_featurizer: nn.Module,
        dataset: Dataset
) -> None:
    for item in dataset:
        # load dataset item
        y, y_dur, out_fp = item
        segment_length = y.shape[-1]

        # run through featurizer
        y = featurizer(y.to(device))

        # run through model
        y_hat = torch.empty_like(y)
        for s, e in get_minibatch(y.shape[0]):
            y_hat[s:e] = model(y[s:e])

        # run through inverse featurizer
        y_hat = inverse_featurizer(y_hat, length=segment_length).cpu()
        # overlap-add moment
        y_hat = overlap_add(y_hat, y_dur, step=dataset.hop_size)

        # delete padded chunks
        y_hat = y_hat[:, dataset.pad_size:-dataset.pad_size]

        # save file as .wav
        sf.write(out_fp, y_hat.T, samplerate=dataset.sr)
    return None


def main(
        in_path: str,
        out_path: str,
        target: str,
        ckpt_path: Optional[str] = None,
) -> None:
    # initialize all modules
    model, featurizer, inverse_featurizer, cfg = initialize_all(target, ckpt_path)
    # initialize dataset
    dataset = InferenceSourceSeparationDataset(in_path, out_path, **cfg.inference_dataset)

    run_inference(model, featurizer, inverse_featurizer, dataset)
    return None


if __name__ == '__main__':
    main(
        args.in_path,
        args.out_path,
        args.target,
        args.ckpt_path,
    )
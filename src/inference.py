import argparse
from pathlib import Path
from omegaconf import OmegaConf
import soundfile as sf
from typing import Optional

import torch

from data import InferenceSourceSeparationDataset
from train import initialize_model, initialize_featurizer
from utils.utils_inference import load_pl_state_dict, get_minibatch, overlap_add


class InferenceProgram:

    SAVED_MODELS_DIR = Path("./saved_models")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(
            self,
            in_file_path: str,
            out_file_path: str,
            target: str,
            ckpt_path: Optional[str] = None,

    ):
        tgt_dir = self.SAVED_MODELS_DIR / target
        self.cfg_path = tgt_dir / 'hparams.yaml'
        self.ckpt_path = next(iter(tgt_dir.glob('*.ckpt'))) if ckpt_path is None else ckpt_path
        self.cfg = OmegaConf.load(self.cfg_path)
        self.cfg.inference_dataset['in_file_path'] = in_file_path
        self.cfg.inference_dataset['out_file_path'] = out_file_path

        # modules
        model, featurizer, inverse_featurizer, dataset = self.initialize_all()
        self.model = model
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer
        self.dataset = dataset

    def initialize_all(self):
        # load featurizer, inverse_featurizer, and model
        featurizer, inverse_featurizer = initialize_featurizer(self.cfg)
        model, *_ = initialize_model(self.cfg)
        # load dataset
        dataset = InferenceSourceSeparationDataset(**self.cfg.inference_dataset)

        # load checkpoint
        state_dict = load_pl_state_dict(self.ckpt_path, device=self.DEVICE)
        _ = model.load_state_dict(state_dict, strict=True)
        _ = model.eval()

        # send all to device
        _ = featurizer.to(self.DEVICE)
        _ = inverse_featurizer.to(self.DEVICE)
        _ = model.to(self.DEVICE)

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

    def inference_all(self) -> None:
        for item in self.dataset:
            # load dataset item
            y, y_dur, out_fp = item
            # run inference on mixture
            y_hat = self.inference_one(y, y_dur)
            # save file as .wav
            sf.write(out_fp, y_hat.T, samplerate=self.dataset.sr)
        return None


def main(
        in_path: str,
        out_path: str,
        target: str,
        ckpt_path: Optional[str] = None,
) -> None:
    program = InferenceProgram(in_path, out_path, target, ckpt_path)
    program.inference_all()
    return None


if __name__ == '__main__':
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

    main(
        args.in_path,
        args.out_path,
        args.target,
        args.ckpt_path,
    )
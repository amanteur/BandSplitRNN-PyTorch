import argparse
import typing as tp
from pathlib import Path

import torch
import soundfile as sf
from omegaconf import OmegaConf

from data import EvalSourceSeparationDataset
from separator import Separator


class InferenceProgram:
    SAVED_MODELS_DIR = Path("./saved_models")

    def __init__(
            self,
            in_path: str,
            out_path: str,
            target: str,
            ckpt_path: tp.Optional[str] = None,
            device: str = 'cuda'
    ):
        self.tgt_dir = self.SAVED_MODELS_DIR / target
        # path to checkpoint
        if ckpt_path is None:
            ckpt_path = self.tgt_dir / f"{target}.pt"
            if not ckpt_path.is_file():
                raise ValueError("{ckpt_path} is missing. Please provide 'ckpt_path' explicitly.")
        self.ckpt_path = ckpt_path

        # config params
        self.cfg_path = self.tgt_dir / 'hparams.yaml'
        self.cfg = OmegaConf.load(self.cfg_path)
        self.cfg.audio_params['in_fp'] = in_path
        self.cfg.audio_params['out_fp'] = out_path

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        )

        # initialize the dataset
        self.dataset = EvalSourceSeparationDataset(mode='inference', **self.cfg.audio_params)

        # initialize the separator
        self.sep = Separator(self.cfg, self.ckpt_path)
        _ = self.sep.eval()
        _ = self.sep.to(self.device)

    def run(self) -> None:
        for (y, out_fp) in self.dataset:
            # send to device
            y = y.to(self.device)

            # apply separator to the mixture file
            y_hat = self.sep(y).cpu()

            # save file as .wav
            sf.write(out_fp, y_hat.T, samplerate=self.dataset.sr)

        return None


def main(args) -> None:
    program = InferenceProgram(**args)
    program.run()
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
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        required=False,
        default='cuda',
        help="Device name - either 'cuda', or 'cpu'."
    )
    args = vars(parser.parse_args())

    main(args)

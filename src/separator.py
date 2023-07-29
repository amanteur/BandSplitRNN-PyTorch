import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig

from train import initialize_model, initialize_featurizer
from utils.utils_inference import load_pl_state_dict, get_minibatch


class Separator(nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
            ckpt_path: tp.Optional[str] = None,
    ):
        super(Separator, self).__init__()
        self.cfg = cfg

        # modules params
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None

        # module initialization
        self.model = self.initialize_modules()

        # audio params
        self.sr = self.cfg.audio_params.sr
        self.chunk_size = int(self.cfg.audio_params.win_size * self.sr)
        self.chunk_step = int(self.cfg.audio_params.hop_size * self.sr)

        # padding for chunk level (used to match stft and istft shapes)
        pad_chunk = self.model[0].win_length - self.chunk_size % self.model[0].hop_length
        self.ws = self.chunk_size + pad_chunk
        self.hs = self.chunk_step + pad_chunk

        # padding for overlap-add
        self.padding_whole = self.chunk_size - self.chunk_step
        self.bs = self.cfg.audio_params.batch_size
        window_name = self.cfg.audio_params.window
        if isinstance(window_name, str):
            self.window = getattr(torch, f'{window_name}_window')(self.ws)
        else:
            self.window = None

    def initialize_modules(self) -> nn.Module:

        # load modules
        featurizer, inverse_featurizer = initialize_featurizer(self.cfg)
        model, *_ = initialize_model(self.cfg)
        _ = model.eval()

        # load checkpoint
        if self.ckpt_path is not None:
            if self.ckpt_path.suffix == '.ckpt':
                state_dict = load_pl_state_dict(self.ckpt_path, device='cpu')
            elif self.ckpt_path.suffix == '.pt':
                state_dict = torch.load(self.ckpt_path, map_location='cpu')
            else:
                raise ValueError(f"Expected checkpoint path, got {self.ckpt_path}.")
            _ = model.load_state_dict(state_dict, strict=True)

        # concat to the one module
        model = nn.Sequential(featurizer, model, inverse_featurizer)

        return model

    def pad(self, y: torch.Tensor) -> torch.Tensor:
        duration = y.shape[-1]
        # padding to divide in even chunks
        padding_add = self.hs - (duration + self.padding_whole * 2 - self.ws) % self.hs
        # pad
        y = F.pad(y, (self.padding_whole, self.padding_whole + padding_add), 'constant')
        return y, padding_add

    def unfold(self, y: torch.Tensor) -> torch.Tensor:
        # unfold
        y = y.unfold(
            -1,
            self.ws,
            self.hs
        ).permute(1, 0, 2)
        return y

    def separate(self, y: torch.Tensor) -> torch.Tensor:
        n_chunks = y.shape[0]
        window = self.window.to(y) if self.window is not None else None

        chunks = []
        for s, e in get_minibatch(n_chunks, self.bs):
            # apply the model
            chunk = self.model(y[s:e])

            if window is None:
                chunk /= (self.ws / self.hs)
            else:
                chunk = chunk * window

            chunks.append(chunk)

        return torch.cat(chunks)

    def fold(
            self,
            y: torch.Tensor,
            n_channels: int,
            duration: int
    ) -> torch.Tensor:
        n_chunks = y.shape[0]
        start = 0

        y_out = torch.zeros((n_channels, duration)).to(y)
        # overlap-add
        for i in range(n_chunks):
            y_out[..., start:start + self.ws] += y[i]
            start += self.hs

        assert start - self.hs + self.ws == duration

        return y_out

    def unpad(
            self,
            y: torch.Tensor,
            padding_add: int
    ) -> torch.Tensor:
        return y[..., self.padding_whole:-(self.padding_whole + padding_add)]

    @torch.no_grad()
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Pads audio, unfolds it into even chunks, applies separation on the chunk level in the TF-domain,
        restores audio via overlap-add.
        Input shape:
            [n_channels, duration]
        Output shape:
            [n_channels, duration]
        """
        # pad signal
        # [n_channels, time]
        y, padding_add = self.pad(y)
        n_channels, duration = y.shape

        # unfold it
        # [n_chunks, n_channels, chunk_size]
        y = self.unfold(y)
        n_chunks = y.shape[0]

        # check shapes
        assert n_chunks == (duration - self.ws) / self.hs + 1

        # apply separation model
        # [n_chunks, n_channels, chunk_size]
        y = self.separate(y)

        # fold back
        # [n_channels, time]
        y = self.fold(y, n_channels, duration)

        # delete padding
        # [n_channels, time]
        y = self.unpad(y, padding_add)

        return y

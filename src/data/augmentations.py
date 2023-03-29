import torch
import torch.nn as nn
import random


class RandomCrop(nn.Module):
    """
    Randomly selects chunk from fragment.
    """

    def __init__(
            self,
            p: float = 1.,
            chunk_size_sec: int = 3,
            window_stft: int = 2048,
            hop_stft: int = 512,
            first_chunk: bool = False,
            sr: int = 44100
    ):
        super().__init__()
        self.p = p

        self.chunk_size = chunk_size_sec * sr
        # additional space to match stft hop size
        pad_chunk = window_stft - self.chunk_size % hop_stft
        self.chunk_size = self.chunk_size + pad_chunk
        self.eval_step = 1 * sr + pad_chunk
        self.first_chunk = first_chunk

    def forward(
            self, y: torch.Tensor
    ) -> torch.Tensor:
        B, S, C, T = y.shape

        if self.training and random.random() < self.p:
            start = random.randrange(0, T - self.chunk_size) if not self.first_chunk else 0
            end = start + self.chunk_size
            y = y[..., start:end]
        if not self.training:
            y = y.unfold(-1, self.chunk_size, self.eval_step)
            n_chunks = y.shape[-2]
            y = y.permute(0, 3, 1, 2, 4).contiguous().view(B * n_chunks, S, C, self.chunk_size)
        return y


class GainScale(nn.Module):
    """
    Randomly scales the energy of a chunk in some dB range.
    """

    def __init__(
            self,
            p: float = 1.,
            min_db: float = -10.,
            max_db: float = 10.,
    ):
        super().__init__()
        self.p = p
        self.min_db = min_db
        self.max_db = max_db

    @staticmethod
    def db2amp(db):
        return 10 ** (db / 20)

    def forward(
            self, y: torch.Tensor
    ) -> torch.Tensor:
        B, S, C, T = y.shape
        device = y.device

        if self.training and random.random() < self.p:
            db_scales = torch.empty(
                B, 1, 1, 1, device=device
            ).uniform_(self.min_db, self.max_db)
            y *= self.db2amp(db_scales)
        return y


class Mix(nn.Module):
    """
    Mixes random target sources into mixtures.
    """

    def __init__(
            self,
            p: float = 0.5,
            min_db: float = 0.,
            max_db: float = 5.,
    ):
        super().__init__()
        self.p = p
        self.min_db = min_db
        self.max_db = max_db

    @staticmethod
    def db2amp(db):
        return 10 ** (db / 20)

    @staticmethod
    def calc_rms(y: torch.Tensor, keepdim=True) -> torch.Tensor:
        """
        Calculate Power of audio signal.
        """
        return torch.sqrt(
            torch.mean(torch.square(y), dim=-1, keepdim=keepdim)
        )

    def rms_normalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Power-normalize an audio signal.
        """
        rms = self.calc_rms(y, keepdim=True)
        return y / (rms + 1e-8)

    def forward(
            self, y: torch.Tensor
    ) -> torch.Tensor:
        B, S, C, T = y.shape
        device = y.device

        if self.training and random.random() < self.p:
            indices_background = torch.randint(
                0, B, (B,),
            )
            db_scales = torch.empty(
                B, 1, 1, 1, device=device
            ).uniform_(self.min_db, self.max_db)
            y_targets_only = y[indices_background, 1].unsqueeze(1).repeat_interleave(2, dim=1)
            y_background = self.rms_normalize(y_targets_only)
            rms_background = self.calc_rms(y) / self.db2amp(db_scales)
            y += y_background * rms_background
        return y

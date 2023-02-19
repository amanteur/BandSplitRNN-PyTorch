import torch
from torch.utils.data import Dataset
import numpy as np
import musdb

from typing import Dict, Union, Tuple
import random

from .preprocessing import SAD


class SourceSeparationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            target: str,
            is_mono: bool,
            musdb_params: Dict[str, Union[bool, str]],
            sad_params: Dict[str, Union[int, float]],
    ):
        """

        :param target:
        :param is_mono:
        :param musdb_params:
        :param sad_params:
        """
        self.target = target
        self.is_mono = is_mono
        self.sr = sad_params['sr']
        self.mus = musdb.DB(**musdb_params)
        self.sad = SAD(**sad_params)

    def prepare_fragments(
            self,
            mix_audio: np.ndarray,
            tgt_audio: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input shape: [n_channels, total_frames]
        Output shape: [n_segments, n_channels, frames_in_segment]
        """
        '''
        
        '''
        mix_audio = torch.tensor(
            mix_audio,
            dtype=torch.float32
        )
        tgt_audio = torch.tensor(
            tgt_audio,
            dtype=torch.float32
        )

        tgt_frags, mask = self.sad(tgt_audio)
        mix_frags, _ = self.sad(mix_audio, mask)

        return (
            mix_frags.transpose(0, 1),
            tgt_frags.transpose(0, 1)
        )

    def __getitem__(self, index):
        """
        Output shape: [n_channels, frames_in_segment]
        """
        track = self.mus[index]
        mix, tgt = self.prepare_fragments(
            track.audio.T,
            track.targets[self.target].audio.T
        )
        segment_idx = random.choice(range(mix.shape[1]))

        mix_segment = mix[segment_idx]
        tgt_segment = tgt[segment_idx]

        if self.is_mono:
            mix_segment = torch.mean(mix_segment, dim=0, keepdim=True)
            tgt_segment = torch.mean(tgt_segment, dim=0, keepdim=True)

        return (
            mix_segment,
            tgt_segment
        )

    def __len__(self):
        return len(self.mus)
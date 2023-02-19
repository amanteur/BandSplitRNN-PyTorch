import torch
import torch.nn as nn
from typing import List, Tuple, Iterator

from .utils import freq2bands

class BandSplitModule(nn.Module):
    """

    """
    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: List[Tuple[int, int]],
            t_timesteps: int = 517,
            fc_dim: int = 128,
    ):
        super(BandSplitModule, self).__init__()

        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.layernorms = nn.ModuleList([
            nn.LayerNorm([1, (e - s), t_timesteps])
            for s, e in self.bandwidth_indices
        ])
        self.fcs = nn.ModuleList([
            nn.Linear((e - s), fc_dim)
            for s, e in self.bandwidth_indices
        ])

    def generate_subband(
            self,
            x: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, 1, freq, time]
        Output: [batch_size, k_subbands, time, fc_output_shape]
        """
        xs = []
        for i, x in enumerate(self.generate_subband(x)):
            x = self.layernorms[i](x)
            x = x.transpose(-1, -2)
            x = self.fcs[i](x)
            xs.append(x)
        return torch.stack(xs, dim=2).squeeze(1)


if __name__ == '__main__':
    batch_size, n_channels, freq, time = 8, 1, 1025, 517
    in_features = torch.rand(batch_size, n_channels, freq, time)

    cfg = {
        "sr": 44100,
        "n_fft": 2048,
        "bandsplits": [
            (1000, 100),
            (4000, 250),
            (8000, 500),
            (16000, 1000),
            (20000, 2000),
        ],
        "t_timesteps": 517,
        "fc_dim": 128
    }

    model = BandSplitModule(**cfg)
    _ = model.eval()

    with torch.no_grad():
        out_features = model(in_features)

    print(f"In: {in_features.shape}\nOut: {out_features.shape}")
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")

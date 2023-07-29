import typing as tp

import torch
import torch.nn as nn

from model.modules.utils import freq2bands


class BandSplitModule(nn.Module):
    """
    BandSplit (1st) Module of BandSplitRNN.
    Separates input in k subbands and runs through LayerNorm+FC layers.
    """

    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            t_timesteps: int = 517,
            fc_dim: int = 128,
            complex_as_channel: bool = True,
            is_mono: bool = False,
    ):
        super(BandSplitModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.layernorms = nn.ModuleList([
            nn.LayerNorm([(e - s) * frequency_mul, t_timesteps])
            for s, e in self.bandwidth_indices
        ])
        self.fcs = nn.ModuleList([
            nn.Linear((e - s) * frequency_mul, fc_dim)
            for s, e in self.bandwidth_indices
        ])

    def generate_subband(
            self,
            x: torch.Tensor
    ) -> tp.Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, n_channels, freq, time]
        Output: [batch_size, k_subbands, time, fc_output_shape]
        """
        xs = []
        for i, x in enumerate(self.generate_subband(x)):
            B, C, F, T = x.shape
            # view complex as channels
            if x.dtype == torch.cfloat:
                x = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
            # from channels to frequency
            x = x.reshape(B, -1, T)
            # run through model
            x = self.layernorms[i](x)
            x = x.transpose(-1, -2)
            x = self.fcs[i](x)
            xs.append(x)
        return torch.stack(xs, dim=1)


if __name__ == '__main__':
    batch_size, n_channels, freq, time = 4, 2, 1025, 259
    is_complex = True
    if is_complex:
        in_features = torch.rand(batch_size, n_channels, freq, time, dtype=torch.cfloat)
    else:
        in_features = torch.rand(batch_size, n_channels, freq, time, dtype=torch.float32)

    cfg = {
        "sr": 44100,
        "complex_as_channel": is_complex,
        "is_mono": n_channels == 1,
        "n_fft": 2048,
        "bandsplits": [
            (1000, 100),
            (4000, 250),
            (8000, 500),
            (16000, 1000),
            (20000, 2000),
        ],
        "t_timesteps": 259,
        "fc_dim": 128
    }

    model = BandSplitModule(**cfg)
    _ = model.eval()

    with torch.no_grad():
        out_features = model(in_features)

    print(f"In: {in_features.shape}\nOut: {out_features.shape}")
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")

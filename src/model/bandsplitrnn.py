import torch
import torch.nn as nn
from typing import List, Tuple

from .modules import BandSplitModule, BandSequenceModelModule, MaskEstimationModule


class BandSplitRNN(nn.Module):
    """
    BandSplitRNN as described in paper.
    """
    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: List[Tuple[int, int]],
            t_timesteps: int,
            fc_dim: int,
            rnn_dim: int,
            rnn_type: str,
            bidirectional: bool,
            num_layers: int,
            mlp_dim: int,
            return_mask: bool=False
        ):
        super(BandSplitRNN, self).__init__()

        self.bandsplit = BandSplitModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim
        )
        self.bandsequence = BandSequenceModelModule(
            k_subbands=len(self.bandsplit.bandwidth_indices),
            t_timesteps=t_timesteps,
            input_dim_size=fc_dim,
            hidden_dim_size=rnn_dim,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            num_layers=num_layers
        )
        self.maskest = MaskEstimationModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            mlp_dim=mlp_dim,
        )
        self.return_mask = return_mask

    def compute_mask(self, x: torch.Tensor):
        """
        Computes complex-valued T-F mask.
        """
        x = self.bandsplit(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.bandsequence(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.maskest(x)  # [batch_size, freq, time]

        return x

    def forward(self, x: torch.Tensor):
        """
        Input and output are T-F complex-valued features.
        Input shape: batch_size, n_channels, freq, time]
        Output shape: batch_size, n_channels, freq, time]
        """
        B, C, F, T = x.shape

        if C > 1:
            x = x.reshape(B*C, 1, F, T)
        # compute T-F mask
        mask = self.compute_mask(x)
        # multiply with original tensor
        x = mask * x

        if C > 1:
            x = x.reshape(B, C, F, T)
            mask = mask.reshape(B, C, F, T)

        if self.return_mask:
            return x, mask
        return x


if __name__ == '__main__':
    batch_size, n_channels, freq, time = 2, 2, 1025, 259
    in_features = torch.rand(batch_size, n_channels, freq, time, dtype=torch.cfloat)
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
        "t_timesteps": 259,
        "fc_dim": 128,
        "rnn_dim": 256,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 1, # 12,
        "mlp_dim": 512,
        "return_mask": False,
    }
    model = BandSplitRNN(**cfg)
    _ = model.eval()

    with torch.no_grad():
        out_features = model(in_features)

    print(model)
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")
    print(f"In: {in_features.shape}\nOut: {out_features.shape}")


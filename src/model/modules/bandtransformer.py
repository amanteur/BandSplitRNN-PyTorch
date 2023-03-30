import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention


class TransformerModule(nn.Module):
    """
    Transformer module based on Dual-Path Transformer paper [1].
    Almost the same as in https://github.com/asteroid-team/asteroid/blob/master/asteroid/masknn/attention.py

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(
            self,
            embed_dim: int = 128,
            dim_ff: int = 512,
            n_heads: int = 4,
            dropout: float = 0.0,
            bidirectional: bool = True,
    ):
        super(TransformerModule, self).__init__()

        self.groupnorm = nn.GroupNorm(embed_dim, embed_dim)
        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.recurrent = nn.LSTM(embed_dim, dim_ff, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(
            2 * dim_ff if bidirectional else dim_ff,
            embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T, across K - keep in mind T->K, K->T

        x = x.view(B * K, T, N)  # [BK, T, N] across T,      [BT, K, N] across K

        # groupnorm
        out = self.groupnorm(
            x.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]

        # Attention
        mha_in = x.transpose(0, 1)
        mha_out, _ = self.mha(mha_in, mha_in, mha_in)
        x = mha_out.transpose(0, 1) + x

        # RNN
        rnn_out, _ = self.recurrent(x)
        x = self.linear(rnn_out) + x

        # returning to the initial shape
        x = x.view(B, K, T, N)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x


class BandTransformerModelModule(nn.Module):
    """
    Modified BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n Transformers in two dimensions - time and subbands.
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            num_layers: int = 6,
    ):
        super(BandTransformerModelModule, self).__init__()

        self.dptransformers = nn.ModuleList([])

        for _ in range(num_layers):
            transformer_across_t = TransformerModule(
                input_dim_size, hidden_dim_size
            )
            transformer_across_k = TransformerModule(
                input_dim_size, hidden_dim_size
            )
            self.dptransformers.append(
                nn.Sequential(transformer_across_t, transformer_across_k)
            )

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        for i in range(len(self.dptransformers)):
            x = self.dptransformers[i](x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, k_subbands, t_timesteps, input_dim = 4, 41, 259, 128
    in_features = torch.rand(batch_size, k_subbands, t_timesteps, input_dim).to(device)

    cfg = {
        "input_dim_size": 128,
        "hidden_dim_size": 256,
        "num_layers": 12
    }
    model = BandTransformerModelModule(**cfg).to(device)
    _ = model.eval()

    with torch.no_grad():
        out_features = model(in_features)

    print(f"In: {in_features.shape}\nOut: {out_features.shape}")
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")

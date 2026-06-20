# %%
import torch
import torch.nn as nn
from Models.Layers.ResidualConv import ResidualBlock
from Models.Layers.SelfAttention import SelfAttention

# %%


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, dropout=0.1):
        super(DownSampleBlock, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock(
                in_channels,
                out_channels,
                stride=2,
                dropout=dropout,
            ),
            SelfAttention(out_channels, dropout=dropout)
            if use_attention
            else nn.Identity(),
        )

    def forward(self, x):
        return self.layers(x)


class UpSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_attention=False,
        dropout=0.1,
    ):
        super(UpSampleBlock, self).__init__()
        self.layers = nn.Sequential(
            SelfAttention(in_channels, dropout=dropout)
            if use_attention
            else nn.Identity(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResidualBlock(in_channels, out_channels, dropout=dropout),
            ResidualBlock(out_channels, out_channels, dropout=dropout),
        )

    def forward(self, x):
        return self.layers(x)


# %%
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels=64,
        num_downsamples=4,
        latent_size=4,
        dropout=0.1,
    ):
        super(Encoder, self).__init__()
        channels = [base_channels * (2**i) for i in range(num_downsamples)]
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                DownSampleBlock(
                    channels[i - 1] if i > 0 else channels[0],
                    channels[i],
                    use_attention=False,
                    dropout=dropout,
                )
                for i in range(num_downsamples)
            ]
        )
        self.mu = nn.Conv2d(
            channels[-1], latent_size, kernel_size=3, stride=1, padding=1
        )
        self.logvar = nn.Conv2d(
            channels[-1], latent_size, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_normal_(self.mu.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(
            self.initial_conv[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.initial_conv[0].bias)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


# %%
class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        base_channels=64,
        num_upsamples=4,
        latent_size=4,
        dropout=0.1,
    ):
        super(Decoder, self).__init__()
        channels = [
            base_channels * (2 ** (num_upsamples - 1 - i)) for i in range(num_upsamples)
        ]
        self.initial_conv = nn.Sequential(
            nn.Conv2d(latent_size, channels[0], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                UpSampleBlock(
                    channels[i - 1] if i > 0 else channels[0],
                    channels[i],
                    use_attention=True if i < 3 else False,
                    dropout=dropout,
                )
                for i in range(num_upsamples)
            ]
        )
        self.final_conv = nn.Conv2d(
            channels[-1], out_channels, kernel_size=3, stride=1, padding=1
        )

        self.activation = nn.Tanh()
        nn.init.kaiming_normal_(
            self.initial_conv[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return self.activation(x)


# %%

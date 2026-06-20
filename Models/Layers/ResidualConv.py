# %%
import torch.nn as nn


# %%
class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, norm_groups=8, kernel=3, stride=1, dropout=0.1
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        have_shortcut = in_channels != out_channels or stride != 1

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.GroupNorm(norm_groups, out_channels),
            )
            if have_shortcut
            else nn.Identity()
        )
        self.activation = nn.SiLU(inplace=True)
        # init
        nn.init.kaiming_normal_(
            self.conv1[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.conv1[0].bias)
        if have_shortcut:
            nn.init.kaiming_normal_(
                self.shortcut[0].weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.zeros_(self.shortcut[0].bias)

        nn.init.zeros_(self.conv2[0].weight)
        nn.init.zeros_(self.conv2[0].bias)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.activation(x + res)

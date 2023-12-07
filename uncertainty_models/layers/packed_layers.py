from torch import nn
from einops import rearrange


class PackedConvLayer(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size, num_estimators, alpha, stride=1, gamma=1, groups=1,
            padding=0, bias=True, first=False, last=False, device=None, dtype=None):

        super().__init__()

        # Input and output channel of each packed conv layer and number of groups per conv
        ensemble_in_channels = int(in_channels * (1 if first else alpha))
        ensemble_out_channels = int(out_channels * (num_estimators if last else alpha))
        conv_groups = 1 if first else gamma * groups * num_estimators

        # checking if the number of channels are divisible by group and
        # if we have minimum 64 channels per group otherwise correcting the values
        while (ensemble_in_channels % conv_groups != 0 or ensemble_in_channels // conv_groups < 64) and conv_groups // (
                groups * num_estimators) > 1:
            gamma -= 1
            conv_groups = gamma * groups * num_estimators

        if ensemble_in_channels % conv_groups:
            ensemble_in_channels += (num_estimators - ensemble_in_channels % conv_groups)

        if ensemble_out_channels % conv_groups:
            ensemble_out_channels += (num_estimators - ensemble_out_channels % conv_groups)

        self.packed_conv = nn.Conv2d(in_channels=ensemble_in_channels, out_channels=ensemble_out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, dilation=1,
                                     groups=conv_groups, bias=bias, padding_mode="zeros", device=device, dtype=dtype
                                     )

    def forward(self, x):
        return self.packed_conv(x)


class PackedLinearLayer(nn.Module):

    def __init__(
            self, in_features, out_features, alpha, num_estimators, gamma=1, bias=True, first=False,
            last=False, device=None, dtype=None):

        super().__init__()

        # Input and output channel of each packed layer and number of groups per layer
        ensemble_in = int(in_features * (1 if first else alpha))
        ensemble_out = int(out_features * (num_estimators if last else alpha))
        groups = num_estimators * gamma if not first else 1

        # checking if the number of channels are divisible by group and
        # otherwise correcting the values
        if ensemble_in % groups:
            ensemble_in += num_estimators - ensemble_in % groups

        if ensemble_out % groups:
            ensemble_out += num_estimators - ensemble_out % groups

        self.linear = nn.Conv1d(in_channels=ensemble_in, out_channels=ensemble_out, kernel_size=1, stride=1, padding=0,
                                dilation=1, groups=groups, bias=bias, padding_mode="zeros", device=device, dtype=dtype)
        self.first = first
        self.num_estimators = num_estimators

    def forward(self, x):
        x = x.unsqueeze(-1)
        if not self.first:
            x = rearrange(x, "(m e) c h -> e (m c) h", m=self.num_estimators)

        x = self.linear(x)
        x = rearrange(x, "e (m c) h -> (m e) c h", m=self.num_estimators).squeeze(-1)
        return x


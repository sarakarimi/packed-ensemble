import torch
from torch import nn, Tensor
from typing import Any, Type, Union
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import optim
from torchmetrics.functional import accuracy


class PackedConvLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        padding_mode: str = "zeros",
        first: bool = False,
        last: bool = False,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = int(in_channels * (1 if first else alpha))
        extended_out_channels = int(
            out_channels * (num_estimators if last else alpha)
        )

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups
            < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        # fix if not divisible by groups
        if extended_in_channels % actual_groups:
            extended_in_channels += (
                num_estimators - extended_in_channels % actual_groups
            )
        if extended_out_channels % actual_groups:
            extended_out_channels += (
                num_estimators - extended_out_channels % actual_groups
            )

        self.packed_conv = nn.Conv2d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.packed_conv(input)


class PackedLinearLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        bias: bool = True,
        rearrange: bool = True,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.first = first
        self.num_estimators = num_estimators
        self.rearrange = rearrange

        # Define the number of features of the underlying convolution

        extended_in_features = int(in_features * (1 if first else alpha))
        extended_out_features = int(
            out_features * (num_estimators if last else alpha)
        )

        # Define the number of groups of the underlying convolution
        actual_groups = num_estimators * gamma if not first else 1

        # fix if not divisible by groups
        if extended_in_features % actual_groups:
            extended_in_features += num_estimators - extended_in_features % (
                actual_groups
            )
        if extended_out_features % actual_groups:
            extended_out_features += num_estimators - extended_out_features % (
                actual_groups
            )

        self.linear = nn.Conv1d(
            in_channels=extended_in_features,
            out_channels=extended_out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=actual_groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def _rearrange_forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        if not self.first:
            x = rearrange(x, "(m e) c h -> e (m c) h", m=self.num_estimators)

        x = self.linear(x)
        x = rearrange(x, "e (m c) h -> (m e) c h", m=self.num_estimators)
        return x.squeeze(-1)

    def forward(self, input: Tensor) -> Tensor:
        if self.rearrange:
            return self._rearrange_forward(input)
        else:
            return self.linear(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        alpha: float = 2,
        num_estimators: int = 4,
        gamma: int = 1,
        groups: int = 1,
    ):
        super(BasicBlock, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConvLayer(
            in_planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConvLayer(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConvLayer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes * alpha),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet50Block(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        alpha: float = 2,
        num_estimators: int = 4,
        gamma: int = 1,
        groups: int = 1,
    ):
        super(Resnet50Block, self).__init__()

        # No subgroups for the first layer
        self.conv1 = PackedConvLayer(
            in_planes,
            planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=1,  # No groups from gamma in the first layer
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConvLayer(
            planes,
            planes,
            kernel_size=3,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * alpha)
        self.conv3 = PackedConvLayer(
            planes,
            self.expansion * planes,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConvLayer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes * alpha),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PackedResNet(LightningModule):
    def __init__(self, arch, in_channels, num_classes, groups, num_estimators, alpha, gamma):
        super().__init__()

        self.arch = arch
        self.in_channels = in_channels
        self.in_planes = 64
        self.alpha = alpha
        self.gamma = gamma
        self.groups = groups
        self.num_estimators = num_estimators

        block_planes = self.in_planes

        # settings of ResNet architectures
        if self.arch == "18":
            block = BasicBlock
            num_blocks = [2, 2, 2, 2]
        elif arch == "50":
            block = Resnet50Block
            num_blocks = [3, 4, 6, 3]
        else:
            raise NotImplemented

        self.conv1 = PackedConvLayer(
            self.in_channels,
            block_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=1,
            groups=groups,
            bias=False,
            first=True,
        )

        self.bn1 = nn.BatchNorm2d(block_planes * alpha)

        self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            groups=groups,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)
        self.linear = PackedLinearLayer(
            block_planes * 8 * block.expansion,
            num_classes,
            alpha=alpha,
            num_estimators=num_estimators,
            last=True,
        )

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Resnet50Block]],
            planes: int,
            num_blocks: int,
            stride: int,
            alpha: float,
            num_estimators: int,
            gamma: int,
            groups: int,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    groups=groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.optional_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = rearrange(
            x, "e (m c) h w -> (m e) c h w", m=self.num_estimators
        )

        x = self.pool(x)
        x = self.flatten(x)
        out = self.linear(x)
        return out

    @staticmethod
    def loss_fn(outputs, targets):
        loss = nn.CrossEntropyLoss()(outputs, targets)
        return loss

    def training_step(self, train_batch, batch_index):
        inputs, targets = train_batch
        targets = targets.repeat(self.num_estimators)
        outputs = self.forward(inputs)
        # loss calculation
        loss = self.loss_fn(outputs, targets)
        return loss

    def evaluate(self, batch, stage=None):
        inputs, targets = batch
        targets = targets.repeat(self.num_estimators)
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        preds = F.softmax(logits, dim=1).mean(dim=1)
        # acc = accuracy(preds, targets, task="multiclass")

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            # self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

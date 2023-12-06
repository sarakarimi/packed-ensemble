from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import optim
from torchmetrics.functional import accuracy


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


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, num_estimators=4, gamma=1, alpha=2, groups=1):
        super(BasicBlock, self).__init__()

        self.conv1 = PackedConvLayer(in_planes, planes, kernel_size=3, alpha=alpha, num_estimators=num_estimators,
                                     groups=groups, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConvLayer(planes, planes, kernel_size=3, alpha=alpha, num_estimators=num_estimators,
                                     gamma=gamma, groups=groups, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * alpha)

        self.res = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.res = nn.Sequential(
                PackedConvLayer(in_planes, planes, kernel_size=1, alpha=alpha, num_estimators=num_estimators,
                                gamma=gamma, groups=groups, stride=stride, bias=False),
                nn.BatchNorm2d(planes * alpha)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.res(x)
        out = F.relu(out)
        return out


class Resnet50Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, alpha=2, num_estimators=4, gamma=1, groups=1):
        super(Resnet50Block, self).__init__()

        self.conv1 = PackedConvLayer(in_planes, planes, kernel_size=1, alpha=alpha, stride=stride,
                                     num_estimators=num_estimators, gamma=1, groups=groups, bias=False)

        self.bn1 = nn.BatchNorm2d(planes * alpha)

        self.conv2 = PackedConvLayer(planes, planes, kernel_size=3, alpha=alpha, num_estimators=num_estimators,
                                     gamma=gamma, stride=stride, groups=groups, bias=False,
                                     )
        self.bn2 = nn.BatchNorm2d(planes * alpha)

        self.conv3 = PackedConvLayer(planes, planes * 4, kernel_size=1, alpha=alpha, num_estimators=num_estimators,
                                     gamma=gamma, groups=groups, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * alpha * 4)

        self.res = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.res = nn.Sequential(
                PackedConvLayer(in_planes, planes * 4, kernel_size=1, alpha=alpha, num_estimators=num_estimators,
                                gamma=gamma, groups=groups, stride=stride, bias=False),
                nn.BatchNorm2d(planes * alpha * 4)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.res(x)
        out = F.relu(out)
        return out


class PackedResNet(LightningModule):
    def __init__(self, arch, in_channels, num_classes, num_estimators, groups=1, alpha=2, gamma=1):
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
            expansion = 1
        elif arch == "50":
            block = Resnet50Block
            num_blocks = [3, 4, 6, 3]
            expansion = 4
        else:
            raise NotImplemented

        self.conv1 = PackedConvLayer(self.in_channels, block_planes, kernel_size=3, stride=1, padding=1, alpha=alpha,
                                     num_estimators=num_estimators, gamma=1, groups=groups, bias=False, first=True)

        self.bn1 = nn.BatchNorm2d(block_planes * alpha)

        self.pooling = nn.Identity()

        self.layer1 = self._make_layer(block, block_planes, num_blocks[0], stride=1, alpha=alpha,
                                       num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.layer2 = self._make_layer(block, block_planes * 2, num_blocks[1], stride=2, alpha=alpha,
                                       num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.layer3 = self._make_layer(block, block_planes * 4, num_blocks[2], stride=2, alpha=alpha,
                                       num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.layer4 = self._make_layer(block, block_planes * 8, num_blocks[3], stride=2, alpha=alpha,
                                       num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)
        self.linear = PackedLinearLayer(block_planes * 8 * expansion, num_classes, alpha=alpha,
                                        num_estimators=num_estimators, last=True)

    def _make_layer(self, block, planes, num_blocks, stride, alpha, num_estimators, gamma, groups, expansion):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, alpha=alpha, num_estimators=num_estimators, gamma=gamma,
                      groups=groups))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pooling(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = rearrange(
            out, "e (m c) h w -> (m e) c h w", m=self.num_estimators
        )

        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
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

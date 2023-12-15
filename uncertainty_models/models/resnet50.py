from torch import nn, Tensor
import torch.nn.functional as F
from uncertainty_models.layers.packed_layers import PackedConvLayer


class Resnet50Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, alpha=2, num_estimators=4, gamma=1, groups=1):
        super(Resnet50Block, self).__init__()

        self.conv1 = PackedConvLayer(in_planes, planes, kernel_size=1, alpha=alpha, num_estimators=num_estimators,
                                     gamma=1, groups=groups, bias=False)

        self.bn1 = nn.BatchNorm2d(planes * alpha)

        self.conv2 = PackedConvLayer(planes, planes, kernel_size=3, padding=1, alpha=alpha,
                                     num_estimators=num_estimators,
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

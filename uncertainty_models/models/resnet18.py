from torch import nn
import torch.nn.functional as F
from uncertainty_models.layers.packed_layers import PackedConvLayer


class Resnet18Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, num_estimators=4, gamma=1, alpha=2, groups=1):
        super(Resnet18Block, self).__init__()

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


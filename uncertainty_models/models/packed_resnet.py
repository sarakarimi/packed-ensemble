import torch
import torchmetrics
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import optim
from uncertainty_models.layers.packed_layers import PackedConvLayer, PackedLinearLayer
from uncertainty_models.utils.metrics import NLL


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


class PackedResNet(LightningModule):
    def __init__(self, arch, in_channels, num_classes, num_estimators, save_milestones, groups=1, alpha=2, gamma=1,
                 lr=0.1, momentum=0.9, weight_decay=5e-4, opt_gamma=0.2, ablation=False, msp=False, ml=False):
        super().__init__()

        self.arch = arch
        self.in_channels = in_channels
        self.in_planes = 64
        self.alpha = alpha
        self.gamma = gamma
        self.groups = groups
        self.num_estimators = num_estimators
        self.save_milestones = save_milestones
        self.lr = lr
        self.momentum = momentum
        self.decay = weight_decay
        self.opt_gamma = opt_gamma
        self.ablation = ablation
        self.msp_criteria = msp
        self.ml_criteria = ml

        block_planes = self.in_planes

        # settings of ResNet architectures
        if self.arch == "18":
            block = Resnet18Block
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

        self.layer1 = self.make_block(block, block_planes, num_blocks[0], stride=1, alpha=alpha,
                                      num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.layer2 = self.make_block(block, block_planes * 2, num_blocks[1], stride=2, alpha=alpha,
                                      num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.layer3 = self.make_block(block, block_planes * 4, num_blocks[2], stride=2, alpha=alpha,
                                      num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.layer4 = self.make_block(block, block_planes * 8, num_blocks[3], stride=2, alpha=alpha,
                                      num_estimators=num_estimators, gamma=gamma, groups=groups, expansion=expansion)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)
        self.linear = PackedLinearLayer(block_planes * 8 * expansion, num_classes, alpha=alpha,
                                        num_estimators=num_estimators, last=True)

        # Tests metrics
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_ece = torchmetrics.classification.CalibrationError(task="multiclass", num_classes=num_classes)
        self.test_aupr = torchmetrics.classification.BinaryAveragePrecision()
        self.test_nll = NLL(reduction="sum")

    def make_block(self, block, planes, num_blocks, stride, alpha, num_estimators, gamma, groups, expansion):
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
        # targets = targets.repeat(self.num_estimators)
        logits = self.forward(inputs)
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)
        preds_probs = F.softmax(logits, dim=-1)
        preds_probs = preds_probs.mean(dim=1)

        acc = self.test_acc(preds_probs, targets)
        ece = self.test_ece(preds_probs, targets)

        if stage == "test":
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_ece", ece, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
        inputs, targets = batch
        logits = self.forward(inputs)
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)
        preds_probs = F.softmax(logits, dim=-1)
        preds_probs = preds_probs.mean(dim=1)

        if self.ablation:
            if self.msp_criteria:
                ood = -preds_probs.max(-1)[0]
            elif self.ml_criteria:
                ood = -logits.mean(dim=1).max(dim=-1)[0]

        self.test_aupr.update(ood, torch.ones_like(targets))
        self.test_nll.update(preds_probs, targets)

    def test_epoch_end(self, outputs):
        self.log(f"test_nll", self.test_nll.compute(), prog_bar=True)
        self.test_nll.reset()

        if self.ablation:
            self.log(f"test_aupr", self.test_aupr.compute(), prog_bar=True)
            self.test_aupr.reset()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.save_milestones, gamma=self.opt_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

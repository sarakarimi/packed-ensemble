import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import optim
from uncertainty_models.layers.packed_layers import PackedConvLayer, PackedLinearLayer
from uncertainty_models.models.resnet18 import Resnet18Block
from uncertainty_models.models.resnet50 import Resnet50Block
from uncertainty_models.utils.metrics import NLL


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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        logits = self.forward(inputs)
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)
        preds_probs = F.softmax(logits, dim=-1)
        preds_probs = preds_probs.mean(dim=1)

        # acc = self.test_acc(preds_probs, targets)
        # ece = self.test_ece(preds_probs, targets)

        if self.ablation:
            if self.msp_criteria:
                ood = -preds_probs.max(-1)[0]
            elif self.ml_criteria:
                ood = -logits.mean(dim=1).max(dim=-1)[0]

        if dataloader_idx == 0:
            self.test_nll.update(preds_probs, targets)
            self.test_ece.update(preds_probs, targets)
            self.test_acc.update(preds_probs, targets)

            if self.ablation:
                self.test_aupr.update(ood, torch.zeros_like(targets))

        elif self.ablation and dataloader_idx == 1:
            self.test_aupr.update(ood, torch.ones_like(targets))

        return logits

    def test_epoch_end(self, outputs):
        self.log(f"test_nll", self.test_nll.compute(), prog_bar=True)
        self.log(f"test_acc", self.test_acc.compute(), prog_bar=True)
        self.log(f"test_ece", self.test_ece.compute(), prog_bar=True)
        self.test_nll.reset()
        self.test_acc.reset()
        self.test_ece.reset()

        if self.ablation:
            self.log(f"test_aupr", self.test_aupr.compute(), prog_bar=True)
            self.test_aupr.reset()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.save_milestones, gamma=self.opt_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

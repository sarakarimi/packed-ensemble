import pytorch_lightning
from uncertainty_models.models.packed_resnet import PackedResNet, BasicBlock
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision
import torch
import os

PATH_DATASETS = "/datasets"
BATCH_SIZE = 128 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

if __name__ == '__main__':
    # dataset
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    arch = "18"
    num_classes = 10
    num_channels = 3
    max_epochs = 1
    groups = 1
    gamma = 2
    alpha = 2
    num_estimators = 4

    # Model
    model = PackedResNet(arch=arch, num_classes=num_classes, in_channels=num_channels,
                         groups=groups, gamma=gamma, alpha=alpha, num_estimators=num_estimators)

    # params = None
    trainer = pytorch_lightning.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

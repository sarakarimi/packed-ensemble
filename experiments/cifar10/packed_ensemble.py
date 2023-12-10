import argparse
import pytorch_lightning
from uncertainty_models.models.packed_resnet import PackedResNet
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision
import os

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=75, help='Number of epochs to run for')
parser.add_argument('--groups', type=int, default=1, help='Number of groups')
parser.add_argument('--gamma', type=int, default=2, help='Number of sub-groups')
parser.add_argument('--alpha', type=int, default=2, help='The width-augmentation factor of Packed-Ensembles')
parser.add_argument('--num_estimator', type=int, default=4, help='The number of subnetworks in an ensemble')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum of optimizer')
parser.add_argument('--decay', type=float, default=5e-4, help='Learning rate weight decay')
parser.add_argument('--opt_gamma', type=float, default=0.2, help='Gamma parameters in the optimizer')
parser.add_argument('--arch', type=str, default="18", help='Resnet architecture, choices are "18" and "50"')

PATH_DATASETS = "/packed-ensemble/datasets"
NUM_WORKERS = int(os.cpu_count() / 2)

if __name__ == '__main__':
    args = parser.parse_args()

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
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    # setting for choice of architecture
    arch = str(args.arch)
    if arch == "18":
        num_classes = 10
        num_channels = 3
        save_milestones = [25, 50]
    elif arch == "50":
        num_classes = 10
        num_channels = 3
        save_milestones = [60, 120, 160]
    else:
        raise NotImplemented

    # hyperparameters
    max_epochs = args.max_epochs
    groups = args.groups
    gamma = args.gamma
    alpha = args.alpha
    num_estimators = args.num_estimator
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    opt_gamma = args.opt_gamma

    # Model
    model = PackedResNet(arch=arch, num_classes=num_classes, in_channels=num_channels, groups=groups, gamma=gamma,
                         alpha=alpha, num_estimators=num_estimators, save_milestones=save_milestones, lr=lr,
                         momentum=momentum, weight_decay=decay, opt_gamma=opt_gamma, ablation=True, ml=True)

    trainer = pytorch_lightning.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)



import argparse

import pytorch_lightning
from uncertainty_models.models.packed_resnet import PackedResNet
from uncertainty_models.utils.data_modules import CIFAR100DataModule
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=75, help='Number of epochs to run for')
parser.add_argument('--groups', type=int, default=1, help='Number of groups')
parser.add_argument('--gamma', type=int, default=2, help='Number of sub-groups')
parser.add_argument('--alpha', type=int, default=2, help='The width-augmentation factor of Packed-Ensembles')
parser.add_argument('--num_estimator', type=int, default=4, help='The number of subnetworks in an ensemble')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum of optimizer')
parser.add_argument('--decay', type=float, default=1e-4, help='Learning rate weight decay')
parser.add_argument('--opt_gamma', type=float, default=0.2, help='Gamma parameters in the optimizer')
parser.add_argument('--arch', type=str, default="18", help='Resnet architecture, choices are "18" and "50"')

PATH_DATASETS = "/datasets"

if __name__ == '__main__':
    args = parser.parse_args()

    # dataset
    train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(15),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
                                           ])

    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
                                          ])

    cifar100_dm = CIFAR100DataModule(
        data_dir=PATH_DATASETS,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )

    # setting for choice of architecture
    arch = str(args.arch)
    if arch == "18":
        num_classes = 10
        num_channels = 3
        save_milestones = [25, 50]
    if arch == "50":
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
                         momentum=momentum, weight_decay=decay, opt_gamma=opt_gamma)

    trainer = pytorch_lightning.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer.fit(model, cifar100_dm)
    trainer.test(model, datamodule=cifar100_dm)

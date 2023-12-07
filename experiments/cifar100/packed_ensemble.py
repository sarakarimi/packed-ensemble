import pytorch_lightning
from uncertainty_models.models.packed_resnet import PackedResNet
from uncertainty_models.utils.data_modules import CIFAR100DataModule
from torchvision import transforms
import torch


PATH_DATASETS = "/datasets"
BATCH_SIZE = 128 if torch.cuda.is_available() else 64

if __name__ == '__main__':
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
        batch_size=BATCH_SIZE,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )

    # arch can be either "18" or "50"
    arch = "18"
    num_classes = 100
    num_channels = 3

    # hyperparameters
    max_epochs = 75
    groups = 1
    gamma = 2
    alpha = 2
    num_estimators = 4

    # Model
    model = PackedResNet(arch=arch, num_classes=num_classes, in_channels=num_channels,
                         groups=groups, gamma=gamma, alpha=alpha, num_estimators=num_estimators)

    trainer = pytorch_lightning.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer.fit(model, cifar100_dm)
    trainer.test(model, datamodule=cifar100_dm)

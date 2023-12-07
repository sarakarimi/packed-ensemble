import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class CIFAR100DataModule(pl.LightningDataModule):
    """
    code taken from : https://github.com/anilbhatt1/emlo_s9_Sagemaker_CIFAR100_4gpu/blob/master/emlo_s9_cifar100_tensorboard_resnet34_v0.ipynb
    """

    def __init__(self, batch_size, train_transforms, test_transforms, data_dir: str = './data'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = train_transforms
        self.transform_test = test_transforms

    def prepare_data(self):
        # download
        torchvision.datasets.CIFAR100(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = torchvision.datasets.CIFAR100(self.data_dir, train=True, transform=self.transform_train)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=16, num_workers=4)


from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import pytorch_lightning as pl


import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl




def init_cifar10dm(data_dir: str = "./", batch_size: int = 32, num_workers: int = 12):

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

    return CIFAR10DataModule(data_dir=data_dir, 
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            train_transforms=train_transforms,
                            test_transforms=train_transforms,
                            val_transforms=test_transforms,
                            )




class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "./", 
                 batch_size: int = 32, 
                 num_workers: int = 12):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    def prepare_data(self):
        # download
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR100(self.data_dir, train=True, transform=self.train_transform)
            size = len(cifar_full)
            self.cifar100_train, self.cifar100_val = random_split(cifar_full, [int(size*0.9), size - int(size*0.9)])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar100_test = datasets.CIFAR100(self.data_dir, train=False, transform=self.test_transform)

        if stage == "predict" or stage is None:
            self.cifar100_predict = datasets.CIFAR100(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.cifar100_predict, batch_size=self.batch_size, num_workers=self.num_workers)
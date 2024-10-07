import os
import zipfile
from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DogBreedDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.zip_path = self.data_dir / "dog-breed-image-dataset.zip"
        self.extract_dir = self.data_dir.joinpath("extracted") 

    def prepare_data(self):
        if not self.extract_dir.exists():
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)

    def setup(self, stage: Optional[str] = None):
        dataset = ImageFolder(root=self.extract_dir.joinpath("dataset"),transform=self.train_transform)
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = int (dataset_size - (train_size + val_size))
        print("ds={0}, train={1}, test={2}, val={2}".format(dataset_size, train_size, test_size, val_size))

        train_ds, val_ds, test_ds = random_split(
                dataset,
                [train_size, val_size, test_size]
        ) 
        print("ds={0}, train={1}, test={2}, val={2}".format(len(dataset), len(train_ds), len(test_ds), len(val_ds)))

        if stage == "fit" or stage is None:
            print("Model fit called!") 
            self.train_dataset, self.val_dataset = train_ds, val_ds
            #self.train_dataset = ImageFolder(root=self.extract_dir.joinpath("dataset/train") , transform=self.train_transform())
            #self.train_dataset = ImageFolder(root=self.extract_dir / "extracted/dataset/*/*_[1-7?]*.jpg", transform=self.train_transform())
            #self.val_dataset = ImageFolder(root=self.extract_dir.joinpath("dataset/valid") , transform=self.val_transform())
            #self.val_dataset = ImageFolder(root=self.extract_dir / "extracted/dataset/*/*_[8-9?]*.jpg", transform=self.val_transform())
        if stage == "test" or stage is None:
            self.test_dataset = test_ds
            #self.test =  random_split(
            #    dataset,
            #    [test_size]
            #)
            #self.test_dataset = ImageFolder(root=self.extract_dir.joinpath("dataset/test") , transform=self.val_transform())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    @property
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

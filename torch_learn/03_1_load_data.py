import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class MyDataset(Dataset):
    def __init__(
        self, annotations_file: str, img_dir: str, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_data = MyDataset(
    annotations_file="data/train_labels.csv",
    img_dir="data/train",
    transform=None,
)
val_data = MyDataset(
    annotations_file="data/val_labels.csv",
    img_dir="data/val",
    transform=None,
)
batch_size = 32

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    num_workers=4,
    shuffle=False,
)

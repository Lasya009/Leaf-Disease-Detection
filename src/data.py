from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class SimpleImageFolderDataset(Dataset):
    def __init__(self, root_dir, classes_file=None, transform=None):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.classes = []
        if classes_file and Path(classes_file).exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.classes = [l.strip() for l in f if l.strip()]
        else:
            self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(p), self.class_to_idx[cls]))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def make_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    train_ds = SimpleImageFolderDataset(Path(data_dir) / 'train', transform=get_transforms(img_size, train=True))
    val_ds = SimpleImageFolderDataset(Path(data_dir) / 'val', classes_file=Path(data_dir) / 'classes.txt', transform=get_transforms(img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes

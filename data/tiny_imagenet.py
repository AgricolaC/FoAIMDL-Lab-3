import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_dir, batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set_full = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_set_from_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=val_transform)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_set_full), generator=generator).tolist()
    train_size = int(0.8 * len(train_set_full))

    train_set = torch.utils.data.Subset(train_set_full, indices[:train_size])
    val_set = torch.utils.data.Subset(val_set_from_train, indices[train_size:])
    
    # Reorganize Val Folder for ImageFolder compatibility
    test_dir = os.path.join(data_dir, 'val')
    img_dir = os.path.join(test_dir, 'images')
    annotations_file = os.path.join(test_dir, 'val_annotations.txt')

    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.split('\t')
                img, cls = parts[0], parts[1]
                new_cls_dir = os.path.join(img_dir, cls)
                os.makedirs(new_cls_dir, exist_ok=True)
                
                img_path = os.path.join(img_dir, img)
                if os.path.exists(img_path):
                    os.rename(img_path, os.path.join(new_cls_dir, img))

    test_set = datasets.ImageFolder(img_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
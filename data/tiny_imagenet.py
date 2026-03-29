import os
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

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    
    # 3. Step 3: Reorganize Val Folder for ImageFolder compatibility
    val_dir = os.path.join(data_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')

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

    val_set = datasets.ImageFolder(img_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import config
from tqdm import tqdm

class BreastCancerDataset(Dataset):
    def __init__(self, hf_dataset, transform, label_col=None, label2id=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        all_cols = hf_dataset.column_names
        self.label_col = label_col or ('label' if 'label' in all_cols else [c for c in all_cols if c != 'image'][0])
        self.label2id = label2id or config.LABEL2ID

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = item["image"].convert("RGB")

        x = self.transform(img) if self.transform is not None else img

        raw_label = item[self.label_col]
        y = torch.tensor(self.label2id[raw_label], dtype=torch.long)

        return {"pixel_values": x, "label": y}

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def create_dataloaders():
    print(f"Cargando dataset desde el Hub: {config.DATASET_NAME}...")
    raw_dataset = load_dataset(config.DATASET_NAME)
    
    if 'test' in raw_dataset:
        train_raw = raw_dataset['train']
        val_raw = raw_dataset['test']
    else:
        split = raw_dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_raw = split['train']
        val_raw = split['test']

    transform = get_transforms()

    train_dataset = BreastCancerDataset(train_raw, get_train_transforms)
    val_dataset = BreastCancerDataset(val_raw, get_val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Dataset listo.")
    return train_loader, val_loader

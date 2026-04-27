import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import config
from tqdm import tqdm

class BreastCancerDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.transform = transform
        self.images = []
        self.labels = []
        
        
        all_cols = hf_dataset.column_names
        label_col = 'label' if 'label' in all_cols else [c for c in all_cols if c != 'image'][0]
        
        unique_labels = sorted(list(set(hf_dataset[label_col])))
        label_map = {original: i for i, original in enumerate(unique_labels)}
        
        print(f"Map de etiquetas: {label_map}")
        if len(unique_labels) > config.NUM_LABELS:
            print(f"El dataset tiene {len(unique_labels)} clases, pero config tiene {config.NUM_LABELS}")

        for item in tqdm(hf_dataset):
            img = item['image'].convert("RGB")
            self.images.append(self.transform(img))
            
            mapped_label = label_map[item[label_col]]
            self.labels.append(torch.tensor(mapped_label, dtype=torch.long))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'pixel_values': self.images[idx],
            'label': self.labels[idx]
        }

def get_transforms():
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

    train_dataset = BreastCancerDataset(train_raw, transform)
    val_dataset = BreastCancerDataset(val_raw, transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Dataset listo.")
    return train_loader, val_loader

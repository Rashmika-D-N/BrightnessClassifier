import os
from PIL import Image
from torch.utils.data import Dataset

class BrightnessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['dark', 'very_dark']
        self.image_paths = []
        self.labels = []
        
        for idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(idx)
                        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

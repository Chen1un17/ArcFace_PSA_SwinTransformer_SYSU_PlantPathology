# src/dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PlantPathologyDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, is_test=False):
        self.labels_frame = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.is_test:
            if self.transform:
                image = self.transform(image)
            return image, self.labels_frame.iloc[idx, 0]  # 返回图像和图像名称
        else:
            labels = self.labels_frame.iloc[idx, 1:].values.astype('float')
            if self.transform:
                image = self.transform(image)
            return image, labels

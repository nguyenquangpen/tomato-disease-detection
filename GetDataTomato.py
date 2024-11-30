"""
@author: Quang Nguyen <nguyenquangpen@gmail.com>
"""

import os
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset

class DataTomato(Dataset):
    def __init__(self, root, train=True, transform=None):

        self.catergories = os.listdir(root)
        print(self.catergories)
        file = os.listdir(root)
        self.label = []
        self.file_path = []

        for label_num, path in enumerate(file):
            images = os.listdir(os.path.join(root, path))
            images = [os.path.join(root, path, image) for image in images]
            image_num = len(images)
            label_x = [label_num] * image_num

            if train:
                self.file_path.extend(images[:int(image_num * 0.8)])
                self.label.extend(label_x[:int(len(label_x) * 0.8)])
            else:
                self.file_path.extend(images[:int(image_num * 0.8)])
                self.label.extend(label_x[:int(len(label_x) * 0.8)])

        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_path[index]

        try:
            image = Image.open(file_path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            print(f"Skipping invalid image file: {file_path}")
            return self.__getitem__(index + 1)
        if self.transform:
            image = self.transform(image)

        return image, self.label[index]

    def __len__(self):
        return len(self.label)

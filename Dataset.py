import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from PIL import Image 
import numpy as np 
import os 

class Dataset_celeba(TensorDataset):
    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size
        self.datasets = os.listdir(path)
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        img = Image.open(self.path + self.datasets[item]).resize([self.image_size, self.image_size])
        return self.transforms(img), np.zeros([0])

    def __len__(self):
        return len(self.datasets)

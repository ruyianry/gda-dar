import numpy as np
import torch
from torch.utils.data import Dataset

class AppendableDataset(Dataset):
    def __init__(self, label_shape):
        self.data = []
        self.labels = []
        self.transform = None
        self.label_shape = label_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # note: since it is only used for classifier, we dont need to convert to one-hot
        # we only use one hot for training the class-conditioned G
        # one_hot = np.zeros(self.label_shape, dtype=np.float32)
        # one_hot[label] = 1
        # label = one_hot

        if self.transform:
            image = self.transform(image)
        image = image.type(torch.FloatTensor)
        return image, label

    def set_transform(self, transform):
        self.transform = transform

    def append(self, img, label):
        self.data.append(img)
        self.labels.append(label)
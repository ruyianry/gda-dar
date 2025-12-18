import numpy as np
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, label_shape):
        self.data = []
        self.labels = []
        self.label_shape = label_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        # convert to one-hot using numpy
        one_hot = np.zeros(self.label_shape, dtype=np.float32)
        one_hot[label] = 1
        image = self.data[idx]

        if self.transform:
            # convert to HWC
            image = image.transpose(1, 2, 0)
            image = self.transform(image) # image now should be tensor

        return image, one_hot, idx

    def append(self, img, label):
        self.data.append(img)
        self.labels.append(label)
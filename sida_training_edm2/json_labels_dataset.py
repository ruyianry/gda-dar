import json
import numpy as np
import torch
from torch.utils.data import Dataset
import dnnlib

class ImageDataset(Dataset):
    def __init__(self, path, resolution=64, name='imagenet512', use_labels=True, num_classes=1000, max_size=None,xflip=False):
        super().__init__()
        self.filepath = path
        self.resolution = resolution
        self.name = name
        self.has_labels = use_labels
        self.num_classes = num_classes
        self.max_size = max_size
        self.xflip=xflip

        # Load data
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        self.labels = data['labels']  # List of [filename, label]

        # Limit dataset size if max_size is set
        if self.max_size is not None and self.max_size < len(self.labels):
            self.labels = self.labels[:self.max_size]

        self.labels_array = np.array([label for _, label in self.labels], dtype=np.int64)
        
        print(f"Loaded {len(self.labels)} labels.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Generate a dummy image tensor
        #image = np.random.normal(size=(8, self.resolution, self.resolution)).astype(np.float32)
        image = np.zeros((8, self.resolution, self.resolution), dtype=np.float32)
        return image, self.get_label(idx) 

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = idx #self.labels_array[idx]
        d.xflip = self.xflip
        d.raw_label = self.labels_array[idx].copy()
        return d
    
    def get_label(self, idx):
        # Generate one-hot encoded label dynamically
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[self.labels_array[idx]] = 1.0
        return label.copy()
        
        # label = self._get_raw_labels()[self._raw_idx[idx]]
        # if label.dtype == np.int64:
        #     onehot = np.zeros(self.label_shape, dtype=np.float32)
        #     onehot[label] = 1
        #     label = onehot
        # return label.copy()

    
    

# import json
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import dnnlib

# class ImageDataset(Dataset):
#     def __init__(self, path, resolution=64, name='imagenet512', use_labels=True, num_classes=1000, max_size=None):
#         super().__init__()
#         self.filepath = path
#         self.resolution = resolution
#         self.name = name
#         self.has_labels = use_labels
#         self.num_classes = num_classes
#         self.max_size = max_size

#         # Load data
#         with open(self.filepath, 'r') as f:
#             data = json.load(f)
#         self.labels = data['labels']  # List of [filename, label]

#         # Limit dataset size if max_size is set
#         if self.max_size is not None and self.max_size < len(self.labels):
#             self.labels = self.labels[:self.max_size]

#         self.labels_array = np.array([label for _, label in self.labels], dtype=np.int64)
#         self.one_hot_labels = np.eye(self.num_classes, dtype=np.float32)[self.labels_array]

#         print(f"Loaded {len(self.labels)} labels.")

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # Return dummy image tensor and corresponding one-hot encoded label
#         return np.random.normal(size=(8, self.resolution, self.resolution)).astype(np.float32), self.one_hot_labels[idx]

#     def get_details(self, idx):
#         d = dnnlib.EasyDict()
#         d.raw_idx = self.labels_array[idx]
#         d.raw_label = self.one_hot_labels[idx]
#         return d


# import json
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import dnnlib

# class ImageDataset(Dataset):
#     def __init__(self, path, resolution=64,name='imagenet512',use_labels=True,num_classes=1000,max_size=None):
#         super().__init__()
#         self.filepath = path
#         self.resolution = resolution
#         self.name = name
#         self.has_labels=use_labels
#         self.num_classes=num_classes
#         self.max_size    = max_size
        
        
        
#         # Load data
#         with open(self.filepath, 'r') as f:
#             data = json.load(f)
#         self.labels = data['labels']  # List of [filename, label]
        
#         self._raw_idx = np.arange(len(self.labels), dtype=np.int64)
        
#         print(f"Loaded {len(self.labels)} labels.")

#         # Prepare label data for efficient access
#         self.labels_array = np.array([label for _, label in self.labels], dtype=np.int64)

#         #self.labels_array =  self.labels_array.astype({1: np.int64, 2: np.float32}[1]) #[labels.ndim])
        
#         self.num_classes = num_classes  # Assuming there are 1000 classes
#         self.one_hot_labels = np.eye(self.num_classes)[self.labels_array]

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         filename, label = self.labels[idx]
#         # For now, we return a dummy tensor for the image and the actual label
#         return np.random.randn(8, 64, 64), self.one_hot_labels[idx]  # Dummy image tensor, real label
    
#     def _open_file(self, fname):
#         # Helper function to open a file, if needed for image loading
#         return open(fname, 'rb')  # Open in binary mode for image files
    
    
# #     def _get_raw_labels(self):
# #         if 1: #elf._raw_labels is None:
# #             self._raw_labels = self.one_hot_labels #self.labels_array #self._load_raw_labels() # if self._use_labels else None
# #             if self._raw_labels is None:
# #                 self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
# #             assert isinstance(self._raw_labels, np.ndarray)
# #             #assert self._raw_labels.shape[0] == self._raw_shape[0]
# #             #assert self._raw_labels.dtype in [np.float32, np.int64]
# #             if self._raw_labels.dtype == np.int64:
# #                 assert self._raw_labels.ndim == 1
# #                 assert np.all(self._raw_labels >= 0)
# #         return self._raw_labels


#     def get_details(self, idx):
#         d = dnnlib.EasyDict()
#         #d.raw_idx = int(self._raw_idx[idx])
#         #d.xflip = (int(self._xflip[idx]) != 0)
#         #d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        
#         d.raw_idx = self.labels_array[idx]
#         d.raw_label = self.one_hot_labels[idx]
#         return d
    
    
    
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

from ..utils.utils import pose_vector_to_matrix
from ..utils.utils import TORCH_DEVICE as device

class ClariusC3HD3Dataset(Dataset):
    
    def __init__(self, filepath):
        hf = h5py.File(filepath)
        self.images = np.array(hf['images'])
        self.images = self.images.astype(np.float32) / 255
        self.images = self.images.transpose((0, 2, 1))

        vecs = np.array(hf['pose_and_wrench'][:, :7])
        self.poses = pose_vector_to_matrix(vecs).astype(np.float32)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).float().to(device)
        pose = torch.from_numpy(self.poses[index]).float().to(device)
        return image, pose
    
    def __len__(self):
        return len(self.images)
    
class SyentheticDataset(Dataset):

    def __init__(self, filepath):
        self.images = np.load(os.path.join(filepath, 'images.npy'))
        self.images = self.images.astype(np.float32) / 255
        self.poses = np.load(os.path.join(filepath, 'poses.npy'))
        self.poses[:, :3, -1] *= 1e-3

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).float().to(device)
        pose = torch.from_numpy(self.poses[index]).float().to(device)
        return image, pose
    
    def __len__(self):
        return len(self.images)
    
class TUESRECDataset(Dataset):
    
    def __init__(self, filepath):
        hf = h5py.File(filepath)
        self.images = np.array(hf['frames'])
        self.images = self.images.astype(np.float32) / 255
        self.poses = np.array(hf['tforms'])

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).float().to(device)
        pose = torch.from_numpy(self.poses[index]).float().to(device)
        return image, pose
    
    def __len__(self):
        return len(self.images)
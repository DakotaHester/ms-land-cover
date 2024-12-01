import torch
import rasterio as rio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from . import transforms as T
from . import utils

from typing import Iterable, Optional, Union, Tuple


class PreTrainDataset(Dataset):
    
    def __init__(self, 
        data_paths: Iterable[str],
        mean: Optional[np.ndarray]=None,
        std: Optional[np.ndarray]=None,
        transform: Optional[transforms.Compose]=T.SimCLRDataAugmentation(),
        return_hsv: bool=False,
        return_metadata: bool=False,
        device: torch.device=torch.device('cpu'),
    ):
        
        self.data_paths = data_paths
        self.transform = transform
        self.return_hsv = return_hsv
        self.return_metadata = return_metadata
        self.device = device
        
        if mean is not None:
            if isinstance(mean, np.ndarray):
                self.mean = torch.tensor(mean, device=device, dtype=torch.float32)
            elif isinstance(mean, torch.Tensor):
                self.mean = mean.float().to(device)
            else:
                raise ValueError(f'mean must be a numpy array or a torch tensor, got {type(mean)}.')
        else:
            self.mean = utils.batched_mean(data_paths, as_tensor=True, device=device)
        
        if std is not None:
            if isinstance(std, np.ndarray):
                self.std = torch.tensor(std, device=device, dtype=torch.float32)
            elif isinstance(std, torch.Tensor):
                self.std = std.float().to(device)
            else:
                raise ValueError(f'std must be a numpy array or a torch tensor, got {type(std)}.')
        else:
            self.std = utils.batched_std(data_paths, mean=self.mean, as_tensor=True, device=device)
    
    
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    
    
    def __getitem__(self, idx: int) -> Union[
        torch.Tensor,                               # image tensor only
        Tuple[torch.Tensor, torch.Tensor],          # image and HSV tensors
        Tuple[torch.Tensor, dict],                  # image tensor and metadata dict
        Tuple[torch.Tensor, torch.Tensor, dict]     # image, HSV tensors, and metadata dict
    ]:
        
        path = self.data_paths[idx]
        img, meta = utils.read_image(
            path, 
            as_float=True, 
            as_tensor=True,
            return_metadata=True, 
            device=self.device,
        )
        img = (img - self.mean) / self.std
        img = self.transform(img)
        
        returns = [img]
        
        if self.return_hsv:
            returns.append(T.rgb_to_hsv(img))
        
        if self.return_metadata:
            returns.append(meta)
        
        if len(returns) == 1: # return only the image tensor
            return returns[0]
        
        return tuple(returns)


class FineTuneDataset(Dataset):
    
    def __init__(self, 
        data_paths: Iterable[str],
        target_paths: Optional[Iterable[str]]=None,
        mean: Optional[np.ndarray]=None,
        std: Optional[np.ndarray]=None,
        transform: Optional[transforms.Compose]=T.StandardDataAugmentations(),
        return_metadata: bool=False,
        device: torch.device=torch.device('cpu'),
    ):
        
        self.data_paths = data_paths
        self.target_paths = target_paths
        self.transform = transform
        self.return_metadata = return_metadata
        self.device = device
        
        if mean is not None:
            if isinstance(mean, np.ndarray):
                self.mean = torch.tensor(mean, device=device, dtype=torch.float32)
            elif isinstance(mean, torch.Tensor):
                self.mean = mean.float().to(device)
            else:
                raise ValueError(f'mean must be a numpy array or a torch tensor, got {type(mean)}.')
        else:
            self.mean = utils.batched_mean(data_paths, as_tensor=True, device=device)
        
        if std is not None:
            if isinstance(std, np.ndarray):
                self.std = torch.tensor(std, device=device, dtype=torch.float32)
            elif isinstance(std, torch.Tensor):
                self.std = std.float().to(device)
            else:
                raise ValueError(f'std must be a numpy array or a torch tensor, got {type(std)}.')
        else:
            self.std = utils.batched_std(data_paths, mean=self.mean, as_tensor=True, device=device)
    
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    
    
    def __getitem__(self, idx: int) -> Union[
        torch.Tensor,                               # image tensor only
        Tuple[torch.Tensor, torch.Tensor],          # image and target tensors
        Tuple[torch.Tensor, dict],                  # image tensor and metadata dict
        Tuple[torch.Tensor, torch.Tensor, dict]     # image, target tensors, and metadata dict
    ]:
        
        path = self.data_paths[idx]
        img, meta = utils.read_image(
            path, 
            as_float=True, 
            as_tensor=True,
            return_metadata=True, 
            device=self.device,
        )
        if self.target_paths is not None:
            target_path = self.target_paths[idx]
            target = utils.read_image(
                target_path, 
                as_float=True, 
                as_tensor=True,
                device=self.device,
            )
            target = target.unsqueeze(0) if len(target.shape) == 2 else target
        img = (img - self.mean) / self.std
        
        if self.target_paths is not None:
            img, target = self.transform(img, target)

        returns = [img]
        
        if self.target_paths is not None:
            returns.append(target)
        
        if self.return_metadata:
            returns.append(meta)
        
        if len(returns) == 1: # return only the image tensor
            return returns[0]
        
        return tuple(returns)
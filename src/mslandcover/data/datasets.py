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
        n_views: int=2,
        mean: Optional[np.ndarray]=None,
        std: Optional[np.ndarray]=None,
        transform: Optional[transforms.Compose]=T.SimCLRDataAugmentation(),
        return_hsv: bool=False,
        return_metadata: bool=False,
        device: torch.device=torch.device('cpu'),
    ):
        
        if return_metadata and isinstance(transform, T.SimCLRDataAugmentation):
            raise ValueError('return_metadata=True is not supported when using SimCLRDataAugmentation.')
        
        self.data_paths = data_paths
        self.n_views = n_views
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
    
    
    def __getitem__(self, idx: int) -> Union[ # many possible return types, hint is not exhaustive (n_views>2 excluded)), 
        torch.Tensor,                                                               # image tensor only
        Tuple[torch.Tensor, torch.Tensor],                                          # (image, hsv) tensors OR (view1, view2) tensors (no hsv or metadata)
        Tuple[Tuple[torch.Tensor, torch.Tensor], dict],                             # (image, hsv) tensors and metadata dict
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] # multiple views of (image, hsv) tensors (n_views=2)
    ]:
        
        path = self.data_paths[idx]
        img, meta = utils.read_image(
            path, 
            as_float=True, 
            as_tensor=True,
            return_metadata=True, 
            device=self.device,
        )
        
        img = img.permute(1, 2, 0)
        img = (img - self.mean) / self.std
        img = img.permute(2, 0, 1)
        
        returns = []
        for _ in range(self.n_views): 
            if self.transform is not None:
                img = self.transform(img)
            
            if self.return_hsv:
                hsv = T.rgb_to_hsv(img)
                returns.append((img, hsv))
            else:
                returns.append(img)
        
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
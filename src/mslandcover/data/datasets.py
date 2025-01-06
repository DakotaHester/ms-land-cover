from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm
import torch
import rasterio as rio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from . import transforms as T
from . import utils
import h5py
import os
from time import time

from typing import Iterable, Optional, Union, Tuple


class PreTrainDataset(Dataset):
    
    def __init__(self, 
        hdf5_path: str,
        hdf5_group: str,
        n_views: int=2,
        data_paths: Optional[Iterable[str]]=None, # Only needed to calculate mean and std, may be removed in the future
        mean: Optional[np.ndarray]=None,
        std: Optional[np.ndarray]=None,
        transform: Optional[transforms.Compose]=T.SimCLRDataAugmentation(),
        return_hsv: bool=False,
        noisy_input: bool=False,
        return_metadata: bool=False,
        device: torch.device=torch.device('cpu'),
        batch_size_for_stats: int=1024,
        n_threads_for_stats: int=os.cpu_count() // 4,
        preload: bool=False,
    ):
        
        if (hdf5_path is not None and hdf5_group is None) or (hdf5_path is None and hdf5_group is not None):
            raise ValueError('Both hdf5_path and hdf5_group must be provided if one is provided.')
        
        if (hdf5_path is None or hdf5_group is None) and data_paths is None:
            raise ValueError('Either hdf5_path and hdf5_group or data_paths must be provided.')

        if return_metadata and hdf5_group is not None:
            raise ValueError('return_metadata=True is not supported when using hdf5 datasets.')
        
        if return_metadata and isinstance(transform, T.SimCLRDataAugmentation):
            raise ValueError('return_metadata=True is not supported when using SimCLRDataAugmentation.')

        # if return_hsv and noisy_input:
            # raise ValueError('Cannot return hsv and noisy input at the same time.')
        
        self.hdf5_path = hdf5_path
        self.hdf5_group = hdf5_group
        self.data_paths = data_paths
        self.n_views = n_views
        self.transform = transform
        self.return_hsv = return_hsv
        self.noisy_input = noisy_input
        self.return_metadata = return_metadata
        self.device = device
        self.preload = preload
        
        if mean is not None:
            if isinstance(mean, np.ndarray):
                self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
            elif isinstance(mean, torch.Tensor):
                self.mean = mean.float().to(device)
            else:
                raise ValueError(f'mean must be a numpy array or a torch tensor, got {type(mean)}.')
        else:
            if self.data_paths is None:
                raise ValueError('mean must be provided if data_paths is not provided.')
            print(f'Computing mean using batch size {batch_size_for_stats} and {n_threads_for_stats} threads.')
            self.mean = utils.batched_mean(
                data=data_paths, 
                batch_size=batch_size_for_stats, 
                as_tensor=True, 
                n_threads=n_threads_for_stats,
                device=device,
            )
        
        if std is not None:
            if isinstance(std, np.ndarray):
                self.std = torch.tensor(std, dtype=torch.float32, device=device)
            elif isinstance(std, torch.Tensor):
                self.std = std.float().to(device)
            else:
                raise ValueError(f'std must be a numpy array or a torch tensor, got {type(std)}.')
        else:
            if self.data_paths is None:
                raise ValueError('std must be provided if data_paths is not provided.')
            print(f'Computing standard deviation using batch size {batch_size_for_stats} and {n_threads_for_stats} threads.')
            self.std = utils.batched_std(
                data=self.data_paths, 
                mean=self.mean, 
                batch_size=batch_size_for_stats, 
                as_tensor=True, 
                n_threads=n_threads_for_stats,
                device=device,
            )
        
        self.data = None
        if self.hdf5_path is not None:
            with h5py.File(self.hdf5_path, 'r') as f:
                self.ids_list = list(f[self.hdf5_group].keys())
                if self.preload:
                    self.data = [torch.from_numpy(f[self.hdf5_group][key][()]) for key in self.ids_list]
        else:
            self.ids_list = [os.path.basename(path).replace('.tif', '') for path in self.data_paths]
            if self.preload:
                worker = partial(utils.read_image, as_float=True, as_tensor=True, device=device)
                pbar = tqdm(total=len(self.data_paths), desc='Preloading data', unit='images')
                with ThreadPoolExecutor(max_workers=n_threads_for_stats) as executor:
                    results = executor.map(worker, self.data_paths)
                    for _ in results:
                        pbar.update(1)
                self.data = list(results)
    
    
    
    def __len__(self) -> int:
        if self.hdf5_path is not None:
            return len(self.ids_list)
        return len(self.data_paths)
    
    
    
    def __getitem__(self, idx: int) -> Union[ # many possible return types, hint is not exhaustive (n_views>2 excluded)), 
        torch.Tensor,                                                               # image tensor only
        Tuple[torch.Tensor, torch.Tensor],                                          # (image, hsv) tensors OR (view1, view2) tensors (no hsv or metadata)
        Tuple[Tuple[torch.Tensor, torch.Tensor], dict],                             # (image, hsv) tensors and metadata dict
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] # multiple views of (image, hsv) tensors (n_views=2)
    ]:
        if self.preload:
            img = self.data[idx]
            
        elif self.hdf5_path is not None:
            key = self.ids_list[idx]
            img = torch.from_numpy(h5py.File(self.hdf5_path, 'r')[self.hdf5_group][key][()])
            
        else:
            path = self.data_paths[idx]
            img, meta = utils.read_image(
                path, 
                as_float=True, 
                as_tensor=True,
                return_metadata=True, 
                device=self.device,
            )
                
        returns = []
        for _ in range(self.n_views): 
            view = self.transform(img) if self.transform is not None else img
            norm_view = T.normalize(view, mean=self.mean, std=self.std)
            
            if self.noisy_input:
                std, lam = torch.rand(1).item() / self.n_views, torch.rand(1).item() / self.n_views
                noisy_view = T.add_noise(norm_view, std=std, lam=lam)
                
            if self.return_hsv:
                hsv = T.rgb_to_hsv(view)
                if self.noisy_input:
                    returns.append((noisy_view.to(self.device), hsv.to(self.device)))
                else:
                    returns.append((norm_view.to(self.device), hsv.to(self.device)))
            else:
                if self.noisy_input:
                    returns.append((noisy_view.to(self.device), norm_view.to(self.device)))
                else:
                    returns.append((norm_view.to(self.device), norm_view.to(self.device)))
                
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
        
        if self.target_paths is not None:
            img, target = self.transform(img, target)

        img = T.normalize(img, mean=self.mean, std=self.std)
        returns = [img]
        
        if self.target_paths is not None:
            returns.append(target)
        
        if self.return_metadata:
            returns.append(meta)
        
        if len(returns) == 1: # return only the image tensor
            return returns[0]
        
        return tuple(returns)

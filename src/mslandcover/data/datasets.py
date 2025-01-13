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

from typing import Iterable, List, Optional, Union, Tuple
from warnings import warn


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
        n_threads_for_stats: int=os.cpu_count(),
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
                    worker = lambda key: torch.from_numpy(f[self.hdf5_group][key][()])
                    pbar = tqdm(total=len(self.ids_list), desc='Preloading data', unit='images')
                    self.data = []
                    with ThreadPoolExecutor(max_workers=n_threads_for_stats) as executor:
                        results = executor.map(worker, self.ids_list)
                        for res in results:
                            self.data.append(res)
                            pbar.update(1)
                    pbar.close()
                        
        else:
            self.ids_list = [os.path.basename(path).replace('.tif', '') for path in self.data_paths]
            if self.preload:
                worker = partial(utils.read_image, as_float=True, as_tensor=True, device=device)
                pbar = tqdm(total=len(self.data_paths), desc='Preloading data', unit='images')
                self.data = []
                with ThreadPoolExecutor(max_workers=n_threads_for_stats) as executor:
                    results = executor.map(worker, self.data_paths)
                    for res in results:
                        self.data.append(res)
                        pbar.update(1)
                pbar.close()
    
    
    
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
                noisy_view = T.add_noise(norm_view, std=std*2, lam=lam*2)
                
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
        preload: bool=True,
        n_threads: int=os.cpu_count()
    ):
        
        self.transform = transform
        self.return_metadata = return_metadata
        self.device = device
        self.preload = preload
                
        # remove files from data paths that do not exist in target paths
        data_path_basenames = [os.path.basename(path) for path in data_paths]
        if target_paths is not None:
            target_path_basenames = [os.path.basename(path) for path in target_paths]
            self.data_paths = [path for path, basename in zip(data_paths, data_path_basenames) if basename in target_path_basenames]
            self.target_paths = [path for path, basename in zip(target_paths, target_path_basenames) if basename in data_path_basenames] 
        else:
            self.data_paths = data_paths
            self.target_paths = None
        
        self.data_paths = sorted(self.data_paths) # sort to ensure that data and target paths match up correctly
        if self.target_paths is not None:
            self.target_paths = sorted(self.target_paths)
        
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
        
        if self.preload:
            worker = partial(utils.read_image, as_float=True, as_tensor=True, device=device)
            pbar = tqdm(total=len(self.data_paths), desc='Preloading data', unit='images')
            self.data = []
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                results = executor.map(worker, self.data_paths)
                for res in results:
                    self.data.append(res)
                    pbar.update(1)
            pbar.close()
            
            if self.target_paths is not None:
                worker = partial(utils.read_image, as_float=False, as_tensor=True, device=device)
                pbar = tqdm(total=len(self.target_paths), desc='Preloading targets', unit='images')
                self.targets = []
                with ThreadPoolExecutor(max_workers=n_threads) as executor:
                    results = executor.map(worker, self.target_paths)
                    for i, res in enumerate(results):
                        res = res.unsqueeze(0) if len(res.shape) == 2 else res
                        if 0 in res:
                            raise ValueError(f'Target image {self.target_paths[i]} contains 0 values. This is not supported.')
                        self.targets.append(res - 1)
                        pbar.update(1)
                pbar.close()    

            self.len = len(self.data)
        else:
            self.len = len(self.data_paths)



    def __len__(self) -> int:
        return self.len
    
    
    
    def __getitem__(self, idx: int) -> Union[
        torch.Tensor,                               # image tensor only
        Tuple[torch.Tensor, torch.Tensor],          # image and target tensors
        Tuple[torch.Tensor, dict],                  # image tensor and metadata dict
        Tuple[torch.Tensor, torch.Tensor, dict]     # image, target tensors, and metadata dict
    ]:
        
        if self.preload:
            img = self.data[idx]
            if self.target_paths is not None:
                target = self.targets[idx]
        else:
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
                    as_float=False, 
                    as_tensor=True,
                    device=self.device,
                )
                if 0 in target:
                    raise ValueError(f'Target image {target_path} contains 0 values. This is not supported.')
                target = target.unsqueeze(0) if len(target.shape) == 2 else target
                target = target - 1

        if self.transform is not None: # color transform expects data in [0, 1] range
            img, target = self.transform(img, target)
        img = T.normalize(img, mean=self.mean, std=self.std) 
        returns = [img]
        
        if self.target_paths is not None:
            returns.append(target.squeeze())
        
        if self.return_metadata:
            returns.append(meta)
        
        if len(returns) == 1: # return only the image tensor
            return returns[0]
        
        return tuple(returns)
    
    
    
    def get_class_distribution(self):
        
        
        if self.target_paths is None:
            raise ValueError('get_class_distribution() is only supported when target_paths are provided.')
        if not self.preload:
            # warn('get_class_distribution() is only supported when preload=True. Returning a constant array')
            # return np.ones(8) / 8
            # raise ValueError('get_class_distribution() is only supported when preload=True.')
            targets_arr = np.concatenate([utils.read_image(path, as_float=False, as_tensor=False) for path in self.target_paths], axis=0).flatten() - 1
            targets_arr = torch.from_numpy(targets_arr)
        else:
            targets_arr = torch.cat(self.targets, dim=0).flatten()
        
        class_counts = torch.bincount(targets_arr, minlength=targets_arr.max() + 1)
        return class_counts.float() / class_counts.sum()
    
    
    
    def oversample_classes(self, class_idxs: List[int], oversample_factor: int=2, minimum_ratio: Optional[List[float]]=None):
        
        for i in range(self.len):
            if self.preload:
                target_tile = self.targets[i]
            else:
                target_tile = utils.read_image(self.target_paths[i], as_float=False, as_tensor=True)
            for j, class_idx in enumerate(class_idxs):
                if class_idx in target_tile:
                    
                    if minimum_ratio is not None:
                        if torch.sum(target_tile == class_idx).item() / target_tile.numel() < minimum_ratio[j]:
                            continue
                    
                    for _ in range(oversample_factor):
                        if self.preload:
                            self.data.append(self.data[i])
                            self.targets.append(target_tile)

                        else:
                            self.target_paths.append(self.target_paths[i])
                            self.data_paths.append(self.data_paths[i])
                        self.len += 1

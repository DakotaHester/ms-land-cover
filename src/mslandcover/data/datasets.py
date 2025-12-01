from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm
import torch
import rasterio as rio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import geopandas as gpd
from . import transforms as T
from . import utils
import h5py
import os
from time import time

from typing import Iterable, List, Optional, Union, Tuple
from warnings import warn


class PreTrainDataset(Dataset):
    
    def __init__(self, 
        hdf5_path: Optional[str]=None,
        hdf5_group: Optional[str]=None,
        n_views: int=2,
        data_paths: Optional[Iterable[str]]=None, # Only needed to calculate mean and std, may be removed in the future
        mean: Optional[np.ndarray]=None,
        std: Optional[np.ndarray]=None,
        transform: Union[Optional[transforms.Compose], List[transforms.Compose]]=T.SimCLRDataAugmentation(),
        return_hsv: bool=False,
        return_lab: bool=False,
        return_spectral_indices: bool=False,
        noisy_input: bool=False,
        noise_std: float=2.0,
        n_bands: int=4,
        # noise_pct: float=0.5,
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
        
        if return_lab and return_hsv:
            raise ValueError('Cannot return both hsv and lab color spaces.')
        
        if sum([return_hsv, return_lab, return_spectral_indices]) > 1:
            raise ValueError('Only one of return_hsv, return_lab, and return_spectral_indices can be True.')

        # if return_hsv and noisy_input:
            # raise ValueError('Cannot return hsv and noisy input at the same time.')
        
        self.hdf5_path = hdf5_path
        self.hdf5_group = hdf5_group
        self.data_paths = data_paths
        self.n_views = n_views
        self.transform = transform
        self.return_hsv = return_hsv
        self.return_lab = return_lab
        self.return_spectral_indices = return_spectral_indices
        self.noisy_input = noisy_input
        self.noise_std = noise_std
        self.n_bands = n_bands
        self.return_metadata = return_metadata
        self.device = device
        self.preload = preload
        
        if isinstance(transform, list) and len(transform) != n_views:
            raise ValueError(f'If transform is a list, it must have length == n_views ({n_views}). Got {len(transform)}.')
        
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
            
            if self.n_bands == 3:
                # create nir composite
                nir_band = img[3, :, :]
                red_band = img[0, :, :]
                green_band = img[1, :, :]
                img = torch.stack([nir_band, red_band, green_band], dim=0)
                
        returns = []
        for i in range(self.n_views):
            if isinstance(self.transform, list):
                view = self.transform[i](img) if self.transform[i] is not None else img 
            else:
                view = self.transform(img) if self.transform is not None else img
            
            norm_view = T.normalize(view, mean=self.mean, std=self.std)
            
            if self.noisy_input:
                noisy_view = T.add_gaussian_noise(norm_view, std=self.noise_std)
                
                # # randomly set some pixels to 0
                # mask = torch.rand_like(norm_view) < self.noise_pct
                # noisy_view[mask] = 0.0
                
            if self.return_hsv:
                hsv = T.rgb_to_hsv(view)
                if self.noisy_input:
                    returns.append((noisy_view.to(self.device), hsv.to(self.device)))
                else:
                    returns.append((norm_view.to(self.device), hsv.to(self.device)))
                    
            elif self.return_lab:
                lab = T.rgb_to_lab(view, contrast_enhance_factor=10)
                if self.noisy_input:
                    returns.append((noisy_view.to(self.device), lab.to(self.device)))
                else:
                    returns.append((norm_view.to(self.device), lab.to(self.device)))
            
            elif self.return_spectral_indices:
                ndvi = T.calculate_ndvi(view)
                ndwi = T.calculate_ndwi(view)
                ngrdi = T.calculate_ngrdi(view)
                si = torch.stack([ndvi, ndwi, ngrdi], dim=0)
                if self.noisy_input:
                    returns.append((noisy_view.to(self.device), si.to(self.device)))
                else:
                    returns.append((norm_view.to(self.device), si.to(self.device)))
            
            else:
                if self.noisy_input:
                    returns.append((noisy_view.to(self.device), noisy_view.to(self.device)))
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
        noise_std: float=0.0, # disable noise by default
        n_bands: int=4,
        transform: Optional[transforms.Compose]=T.StandardDataAugmentations(),
        return_metadata: bool=False,
        device: torch.device=torch.device('cpu'),
        preload: bool=True,
        n_threads: int=os.cpu_count()
    ):
        
        self.n_bands = n_bands
        self.transform = transform
        self.return_metadata = return_metadata
        self.device = device
        self.preload = preload
        self.noise_std = noise_std
                
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
            # if n_bands == 3 and self.mean.shape[0] == 4:
                # use color infrared composite (4, 1, 2)
                # self.mean = torch.tensor([self.mean[3], self.mean[0], self.mean[1]], device=device, dtype=torch.float32)
        
        if std is not None:
            if isinstance(std, np.ndarray):
                self.std = torch.tensor(std, device=device, dtype=torch.float32)
            elif isinstance(std, torch.Tensor):
                self.std = std.float().to(device)
            else:
                raise ValueError(f'std must be a numpy array or a torch tensor, got {type(std)}.')
        else:
            self.std = utils.batched_std(data_paths, mean=self.mean, as_tensor=True, device=device)
            # if n_bands == 3 and self.std.shape[0] == 4:
                # use color infrared composite (4, 1, 2)
                # self.std = torch.tensor([self.std[3], self.std[0], self.std[1]], device=device, dtype=torch.float32)
        
        if (mean is None or std is None) and self.n_bands == 3 and self.mean.shape[0] == 4:
            # use color infrared composite (4, 1, 2)
            self.mean = torch.tensor([self.mean[3], self.mean[0], self.mean[1]], device=device, dtype=torch.float32)
            self.std = torch.tensor([self.std[3], self.std[0], self.std[1]], device=device, dtype=torch.float32)
        
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

        if self.n_bands == 3:
            # create nir composite
            nir_band = img[3, :, :]
            red_band = img[0, :, :]
            green_band = img[1, :, :]
            img = torch.stack([nir_band, red_band, green_band], dim=0)
            
        if self.transform is not None: # color transform expects data in [0, 1] range
            img, target = self.transform(img, target)
        
        img = T.normalize(img, mean=self.mean, std=self.std) 
        if self.noise_std > 0:
            img = T.add_gaussian_noise(img, std=self.noise_std)
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




class TestDataset(Dataset):
    def __init__(self, points_gdf: gpd.GeoDataFrame, raster_paths: List[str], n_bands: int = 3, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.raster_paths = raster_paths
        self.n_bands = n_bands
        self.mean = mean
        self.std = std
        
        self.points_gdf = points_gdf.loc[points_gdf['ground_truth'] != 0]
    
    def __len__(self):
        return len(self.points_gdf)
    
    def __getitem__(self, idx):
        
        point = self.points_gdf.iloc[idx]
        
        raster_id = point['id']
        raster_path = [path for path in self.raster_paths if raster_id + '.tif' == os.path.basename(path)]
        
        assert len(raster_path) == 1, f"Raster path not found for point index {idx}"
        raster_path = raster_path[0]
        
        img, meta = utils.read_image(raster_path, as_float=True, as_tensor=True, return_metadata=True)
        
        if self.n_bands == 3:
            nir_band = img[3, :, :]
            red_band = img[0, :, :]
            green_band = img[1, :, :]
            img = torch.stack([nir_band, red_band, green_band], dim=0)
        
        if self.mean is not None and self.std is not None:
            img = T.normalize(img, mean=self.mean, std=self.std)

        # print(point)
        # print(point['geometry']
        x, y = point.geometry.x, point.geometry.y
        # row, col = meta['transform'].index(x, y, op='round')
        row, col = rio.transform.rowcol(meta['transform'], x, y)
        row, col = int(row), int(col)
        
        class_idx = point['ground_truth']
        class_name = point['ground_truth_class_name']
        
        return_dict = {
            'image': img,
            'row': row,
            'col': col,
            'class_idx': class_idx,
            'class_name': class_name,
            'point_id': point['id'],
        }
        for key, value in return_dict.items():
            if value is None:
                raise ValueError(f"Missing value for key: {key} in point index {idx}")
        return return_dict

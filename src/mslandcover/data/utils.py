import rasterio as rio
import numpy as np
import torch
from math import ceil
import torch

from typing import Iterable, Union, Tuple

def read_image(
    path: str, 
    as_float: bool=True,
    as_tensor: bool=False, 
    return_metadata: bool=False,
    device: torch.device=torch.device('cpu'),
) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, dict], Tuple[torch.Tensor, dict]]:
    
    with rio.open(path) as src:
        img = src.read().squeeze()
        if return_metadata:
            meta = dict(src.meta)
    
    if as_float:
        img = img.astype(np.float32) / 255.0
    
    if as_tensor:
        dtype = torch.float32 if as_float else torch.uint8
        img = torch.tensor(img, device=device, dtype=dtype)
    
    if return_metadata:
        return img, meta
    return img



# algorithm for calculating mean of data without loading all data into memory
# adapted from https://stackoverflow.com/a/75496541
def batched_mean(
    data_paths: Iterable[str], 
    batch_size: int=2048,
    as_tensor: bool=False,
    device: torch.device=torch.device('cpu'),
) -> Union[np.ndarray, torch.Tensor]:
    
    N = len(data_paths)
    num_steps = ceil(N / batch_size)
    mean = np.zeros(3, dtype=np.float64)
    
    for i in range(num_steps):
        start = i * batch_size
        end = min(start + batch_size, N)
        batch = data_paths[start:end]
        images = np.array(
            [read_image(path) for path in batch], 
            dtype=np.float64,
        )
        mean += ((images.mean(axis=(0, 2, 3)) * len(batch)) / N)
    
    mean = mean.astype(np.float32)
    if as_tensor:
        return torch.tensor(mean, device=device, dtype=torch.float32)

    return mean
 
 

def batched_std(
    data_paths: Iterable[str], 
    mean: np.ndarray, 
    batch_size: int=2048,
    as_tensor: bool=False,
    device: torch.device=torch.device('cpu'),
) -> Union[float, int]:
    
    if type(mean) is torch.Tensor:
        mean = mean.cpu().numpy()
    
    N = len(data_paths)
    num_steps = ceil(N / batch_size)
    std = np.zeros(3)
    
    for i in range(num_steps):
        start = i * batch_size
        end = min(start + batch_size, N)
        batch = data_paths[start:end]
        images = np.array(
            [read_image(path) for path in batch],
            dtype=np.float64,
        ).reshape(-1, 256, 256, 3) # need to reshape to (N, H, W, C) for broadcasting with mean
        # similar approach to mean calculation - need per-image sum of squares such that N is meaningful
        std += (np.mean((images - mean) ** 2, axis=(0, 1, 2)) * len(batch)) / N

    std = np.sqrt(std).astype(np.float32)
    if as_tensor:
        return torch.tensor(std, device=device, dtype=torch.float32, )
    
    return std
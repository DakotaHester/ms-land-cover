import rasterio as rio
import numpy as np
import torch
from math import ceil
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import h5py

from typing import Iterable, Union, Tuple

def read_image(
    path: str, 
    as_float: bool=True,
    as_tensor: bool=False, 
    return_metadata: bool=False,
    device: torch.device=torch.device('cpu'),
) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, dict], Tuple[torch.Tensor, dict]]:
    
    with rio.open(path) as src:
        if as_tensor:
            img = torch.from_numpy(src.read()).to(device)
        else: 
            img = src.read()
        if return_metadata:
            meta = dict(src.meta)
    
    if as_float:
        if as_tensor:
            img = img.float() / 255.0
        else: 
            img = img.astype(np.float32) / 255.0
    
    if return_metadata:
        return img, meta
    return img



def read_images(
    paths: Iterable[str], 
    n_threads: int=8, 
    as_tensor: bool=False,
    device: torch.device=torch.device('cpu'),
    as_float: bool=True,
) -> Union[np.ndarray, torch.Tensor]:
    
    func = partial(read_image, as_float=as_float, as_tensor=as_tensor, device=device, return_metadata=False)
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        if not as_tensor:
            return np.array(list(executor.map(func, paths)), dtype=np.float32)
        else:
            return torch.stack(list(executor.map(func, paths)), dim=0)


# algorithm for calculating mean of data without loading all data into memory
# adapted from https://stackoverflow.com/a/75496541
def batched_mean(
    data: Union[Iterable[str], h5py.Group], # iterable of paths or HDF5 group 
    batch_size: int=4096,
    as_tensor: bool=False,
    device: torch.device=torch.device('cpu'),
    n_threads: int=8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the mean of data in batches without loading all data into memory.

    This function computes the mean of a dataset by processing it in batches,
    which is useful for large datasets that cannot fit into memory. The data
    can be provided as an iterable of file paths or an HDF5 group.

    Parameters
    ----------
    data : Union[Iterable[str], h5py.Group]
        An iterable of file paths or an HDF5 group containing the data.
    batch_size : int, optional
        The number of samples to process in each batch, by default 4096.
    as_tensor : bool, optional
        If True, the mean is returned as a torch.Tensor. Otherwise, it is returned
        as a numpy.ndarray, by default False.
    device : torch.device, optional
        The device on which to perform the computation if `as_tensor` is True,
        by default torch.device('cpu').
    n_threads : int, optional
        The number of threads to use for parallel processing, by default 8.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The mean of the data. The type of the return value depends on the `as_tensor`
        parameter.

    Notes
    -----
    This function is adapted from https://stackoverflow.com/a/75496541.

    Examples
    --------
    >>> import h5py
    >>> data = h5py.File('data.h5', 'r')['images']
    >>> mean = batched_mean(data, batch_size=1024, as_tensor=True, device=torch.device('cuda'))
    >>> print(mean)
    tensor([0.4850, 0.4560, 0.4060], device='cuda:0')
    """
    
    data_paths = data.keys() if isinstance(data, h5py.Group) else data
    
    N = len(data_paths)
    num_steps = ceil(N / batch_size)
    if as_tensor:
        mean = torch.zeros(3, device=device, dtype=torch.float32)
    else:
        mean = np.zeros(3, dtype=np.float64)
    
    for i in range(num_steps):
        start = i * batch_size
        end = min(start + batch_size, N)
        batch = data_paths[start:end]
        
        if isinstance(data, h5py.Group):
            if as_tensor:
                images = torch.stack([
                    torch.from_numpy(data[path][()]).to(device) \
                    for path in batch
                ])
            else:
                images = np.stack([data[path][()] for path in batch])
    
        else:
            images = read_images(batch, n_threads=n_threads, as_tensor=as_tensor, device=device)
            
        if as_tensor:
            mean += (torch.mean(images, axis=(0, 2, 3)) * len(batch)) / N
        else:
            mean += ((images.mean(axis=(0, 2, 3)) * len(batch)) / N)

    if as_tensor:
        return mean.float()
    else:
        return mean.astype(np.float32)
 
 


def batched_std(
    data: Union[Iterable[str], h5py.Group],
    mean: Union[np.ndarray, torch.Tensor],
    batch_size: int=4096,
    as_tensor: bool=False,
    device: torch.device=torch.device('cpu'),
    n_threads: int=8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the standard deviation of data in batches without loading all data into memory.

    This function computes the standard deviation of a dataset by processing it in batches,
    which is useful for large datasets that cannot fit into memory. The data
    can be provided as an iterable of file paths or an HDF5 group.

    Parameters
    ----------
    data : Union[Iterable[str], h5py.Group]
        An iterable of file paths or an HDF5 group containing the data.
    mean : Union[np.ndarray, torch.Tensor]
        The mean of the data.
    batch_size : int, optional
        The number of samples to process in each batch, by default 4096.
    as_tensor : bool, optional
        If True, the standard deviation is returned as a torch.Tensor. Otherwise, it is returned
        as a numpy.ndarray, by default False.
    device : torch.device, optional
        The device on which to perform the computation if `as_tensor` is True,
        by default torch.device('cpu').
    n_threads : int, optional
        The number of threads to use for parallel processing, by default 8.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The standard deviation of the data. The type of the return value depends on the `as_tensor`
        parameter.

    Notes
    -----
    This function is adapted from https://stackoverflow.com/a/75496541.

    Examples
    --------
    >>> import h5py
    >>> data = h5py.File('data.h5', 'r')['images']
    >>> mean = batched_mean(data, batch_size=1024, as_tensor=True, device=torch.device('cuda'))
    >>> std = batched_std(data, mean, batch_size=1024, as_tensor=True, device=torch.device('cuda'))
    >>> print(std)
    tensor([0.2290, 0.2240, 0.2250], device='cuda:0')
    """
    
    data_paths = data.keys() if isinstance(data, h5py.Group) else data
    
    if isinstance(mean, torch.Tensor) and not as_tensor:
        mean = mean.detach().cpu().numpy()    
    
    if isinstance(mean, np.ndarray) and as_tensor:
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
    
    N = len(data_paths)
    num_steps = ceil(N / batch_size)
    if as_tensor:
        std = torch.zeros(3, device=device, dtype=torch.float32)
    else:
        std = np.zeros(3)
    
    for i in range(num_steps):
        start = i * batch_size
        end = min(start + batch_size, N)
        batch = data_paths[start:end]
        
        if isinstance(data, h5py.Group):
            if as_tensor:
                images = torch.stack([
                    torch.from_numpy(data[path][()]).to(device) \
                    for path in batch
                ])
            else:
                images = np.stack([data[path][()] for path in batch])
        
        else:
            images = read_images(batch, n_threads=n_threads, as_tensor=as_tensor, device=device)
        
        if as_tensor:
            images = images.permute(0, 2, 3, 1)
        else:
            images = images.transpose(0, 2, 3, 1)
        
        if as_tensor:
            std += (torch.mean((images - mean) ** 2, dim=(0, 1, 2)) * len(batch)) / N
        else:
            std += (np.mean((images - mean) ** 2, axis=(0, 1, 2)) * len(batch)) / N

    if as_tensor:
        return torch.sqrt(std).to(device)
    return np.sqrt(std).astype(np.float32)



def batched_min_max(
    data_paths: Iterable[str],  
    batch_size: int=2048,
    as_tensor: bool=False,
    device: torch.device=torch.device('cpu'),
) -> Union[int, int]:
    
    N = len(data_paths)
    num_steps = ceil(N / batch_size)
    min_val = np.array([np.inf, np.inf, np.inf])
    max_val = np.array([-np.inf, -np.inf, -np.inf])
    
    for i in range(num_steps):
        start = i * batch_size
        end = min(start + batch_size, N)
        batch = data_paths[start:end]
        images = np.array(
            [read_image(path) for path in batch],
            dtype=np.float64,
        ).reshape(-1, 256, 256, 3)
        min_val = np.minimum(min_val, np.min(images, axis=(0, 1, 2)))
        max_val = np.maximum(max_val, np.max(images, axis=(0, 1, 2)))
    
    if as_tensor:
        return torch.tensor(min_val, device=device, dtype=torch.float32), \
            torch.tensor(max_val, device=device, dtype=torch.float32)
    
    return min_val, max_val

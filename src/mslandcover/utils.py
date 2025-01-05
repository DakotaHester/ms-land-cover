import os
from typing import List, Optional, Tuple
import pandas as pd
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.amp import autocast
from .gradcaching import cached, cat_input_tensor
import datetime

from typing import Callable, Dict, Union, Tuple

def get_torch_device() -> torch.device:
    '''
    Get the torch device to use for training.
    
    Returns
    -------
    torch.device
        The torch device.
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def raise_if_not_exists(path: str) -> None:
    '''
    Raise a FileNotFoundError if the path does not exist.
    
    Parameters
    ----------
    path : str
        The path to check for existence.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path {path} does not exist.')



def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    '''
    Convert a hex color to an RGB tuple.
    
    Parameters
    ----------
    hex_color : str
        The hex color to convert.
    
    Returns
    -------
    Tuple[int, int, int]
        The RGB tuple.
    '''
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))



def nt_xent_loss(z_0: torch.Tensor, z_1: torch.Tensor, temperature: float=0.5, reduction='sum') -> torch.Tensor:
    """
    Functional implementation of the normalized temperature-scaled cross entropy loss.

    Parameters
    ----------
    z_0 : torch.Tensor
        The embeddings for the first view of the batch.
    z_1 : torch.Tensor
        The embeddings for the second view of the batch.
    temperature : float
        The temperature scaling factor.
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns
    -------
    torch.Tensor
        The loss value.

    Raises
    ------
    ValueError
        If the input tensors are not of rank 2 or have mismatched sizes.
    """
    if len(z_0.size()) != 2 or len(z_1.size()) != 2:
        raise ValueError("Input tensors must be of rank 2.")
    if z_0.size() != z_1.size():
        raise ValueError("Input tensors must have the same shape.")

    # Concatenate embeddings along the batch dimension
    x = torch.cat([z_0, z_1], dim=0)

    # L2 normalize the input tensor
    x = F.normalize(x, p=2, dim=1)

    # Cosine similarity
    xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
    xcs[torch.eye(x.size(0), device=x.device).bool()] = float("-inf")

    # Ground truth labels
    batch_size = z_0.size(0)
    target = torch.arange(2 * batch_size, device=x.device)
    target[:batch_size] += batch_size  # Map view 1 to view 2
    target[batch_size:] -= batch_size  # Map view 2 to view 1

    # Standard cross entropy loss
    return F.cross_entropy(xcs / temperature, target, reduction=reduction)



# NOTE: Due to the size of the inputs passed to MSE for reconstruction compared to
# NT-Xent for cotnrastive learning, the magnitude of the loss will vary significantly.
# In order to make sure that the NT-Xent loss is not overwhelmed by the MSE loss,
# both losses will be normalized by the number of elements in the input tensor.
# (Not accounting for the batch size, as the batch size is the same for both losses.)
def normalized_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str='sum') -> torch.Tensor:
    return F.mse_loss(y_pred, y_true, reduction=reduction) / y_pred[0,:].numel()



def normalized_nt_xent_loss(z: torch.Tensor) -> torch.Tensor:
    return nt_xent_loss(z) / z[0,:].numel()



@cached
@autocast(get_torch_device().type)
def cached_model_call(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    return model(X)



@cat_input_tensor
@autocast(get_torch_device().type)
def cached_contrastive_loss_call(z_0: torch.Tensor, z_1: torch.Tensor) -> torch.Tensor:
    return nt_xent_loss(z_0, z_1).to(z_0.device)



@cat_input_tensor
@autocast(get_torch_device().type)
def cached_mse_loss_call(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true, reduction='sum')



def init_grad_cache_closure_dicts(n_views: int=2, cache_contents: str=['z', 'y', 'y_hat']) -> Tuple[List[Dict[str, List]], List[List]]:
    cache = [{content: [] for content in cache_contents} for _ in range(n_views)]
    closures = [[] for _ in range(n_views)] # only one closure foe each view
    
    return cache, closures



def call_closures(
    cache: List[Dict[str, List[torch.Tensor]]],
    closures: List[Callable[[torch.Tensor], None]],
    ignore_keys: List[str]=['y'], # ignore the y key in the cache - no gradients to compute 
) -> None:
    if len(cache) != len(closures):
        raise ValueError('The number number of elements in both cache and closures must be the same - check to make sure the views are consistent.')
    n_views = len(cache)
    for i in range(n_views):
        for closure_fn in closures[i]:
            for key in cache[i].keys():
                if key in ignore_keys:
                    continue
                closure_fn(cache[i][key])



class ProfilerHistory:
    
    def __init__(self, device: torch.device):
        
        self.device = device
        self.profiler_history_dict = {
            'epoch': [],
            'phase': [],
            'step': [],
            'time': [],
            'mem_usage': [],
            'mem_alloc': [],
            'mem_cache': [],
            'power_draw': [],
            'gpu_util': [],
            'temperature': [],
            'notes': [],
        }
    
    def update(self, epoch: int, phase: str, step: int, time: int, notes: Optional[str]=None) -> None:
        
        self.profiler_history_dict['epoch'].append(epoch)
        self.profiler_history_dict['phase'].append(phase)
        self.profiler_history_dict['step'].append(step)
        self.profiler_history_dict['time'].append(time)
        
        try:
            for key, func in [
                ('mem_usage', torch.cuda.memory_usage),
                ('mem_alloc', torch.cuda.memory_allocated),
                ('mem_cache', torch.cuda.memory_reserved),
                ('power_draw', torch.cuda.power_draw),
                ('gpu_util', torch.cuda.utilization),
                ('temperature', torch.cuda.temperature),
            ]:
                try:
                    self.profiler_history_dict[key].append(func(self.device))
                except:
                    self.profiler_history_dict[key].append(-1)
        except:
            for key in ['mem_usage', 'mem_alloc', 'mem_cache', 'power_draw', 'gpu_util', 'temperature']:
                self.profiler_history_dict[key].append(-1)
        
        
            self.profiler_history_dict['notes'].append(notes if notes is not None else '')
        
    
    def save(self, path: str) -> None:
        
        n_rows = len(self.profiler_history_dict['epoch'])
        df = pd.DataFrame(self.profiler_history_dict, index=list(range(n_rows)))
        df.to_csv(path, index=False)
    
    
    
    def load(self, path: str) -> None:
        df = pd.read_csv(path)
        self.profiler_history_dict = df.to_dict(orient='list')



def get_datetime(surrounding_brackets: bool=True) -> str:
    
    dt = datetime.datetime.now(datetime.timezone.utc)
    if surrounding_brackets:
        return '[' + dt.strftime('%Y-%m-%d %H:%M:%SZ') + ']'
    return dt.strftime('%Y-%m-%d %H:%M:%SZ')


class Logger:
    def __init__(self, path: str, exist_ok: bool=True):
        self.path = path
        if os.path.exists(path):
            if not exist_ok:
                raise FileExistsError(f'The file {path} already exists.')
            os.remove(path)
    
    def write(self, message: str) -> None:
        with open(self.path, 'a') as f:
            f.write(message + '\n')
    
    def log(self, message: str, prepend_timestamp: bool=True, echo: bool=True) -> None:
        if prepend_timestamp:
            message = get_datetime() + ' ' + message
        if echo: 
            print(message)
        self.write(message)

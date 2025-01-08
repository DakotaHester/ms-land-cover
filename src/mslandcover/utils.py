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

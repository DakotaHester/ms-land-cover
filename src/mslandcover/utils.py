import os
from typing import Tuple
import pandas as pd
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer, required
import torch
import torch.nn as nn
from .gradcaching import cached, cat_input_tensor
import datetime

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


class NTXentLoss(nn.Module):
    """
    Normalized temperature-scaled cross entropy loss for self-supervised learning.
    Code adapted from: https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
    """
    
    def __init__(self, temperature: float=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        assert len(x.size()) == 2
        
        # Cosine similarity
        xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
        xcs[torch.eye(x.size(0)).bool()] = float("-inf")
        
        # Ground truth labels
        target = torch.arange(8)
        target[0::2] += 1
        target[1::2] -= 1
        
        # Standard cross entropy loss
        return F.cross_entropy(xcs / self.temperature, target, reduction="mean")



def nt_xent_loss(x: torch.Tensor, temperature: float=0.5, reduction: str='sum') -> torch.Tensor:
    """
    Functional implementation of the normalized temperature-scaled cross entropy loss.
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    temperature : float
        The temperature scaling factor.
    
    Returns
    -------
    torch.Tensor
        The loss value.
    
    Raises
    ------
    ValueError
        If the input tensor is not of rank 2.
    """
    
    if len(x.size()) != 2:
        raise ValueError(f'Expected input tensor of rank 2, got tensor of rank {len(x.size())}.')
    
    # Cosine similarity
    xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    xcs[torch.eye(x.size(0), device=x.device).bool()] = float("-inf")
    
    # Ground truth labels
    target = torch.arange(len(x), device=x.device) # len(X) to match batch size of input
    target[0::2] += 1
    target[1::2] -= 1
        
    # Standard cross entropy loss
    return F.cross_entropy(xcs / temperature, target, reduction=reduction)



class HybridLoss(nn.Module):
    
    def __init__(self, mse_weight: float=0.5):
        super(HybridLoss, self).__init__()
        self.nt_xent_loss = NTXentLoss()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
    
    def forward(self, 
        z_1: torch.Tensor, z_2: torch.Tensor, # embeddings from the projection head
        y_pred_1: torch.Tensor, y_pred_2: torch.Tensor, # predictions from the image decoder head
        y_true_1: torch.Tensor, y_true_2: torch.Tensor, # target images
    ) -> torch.Tensor:
        contrastive_loss = self.nt_xent_loss(z_1) 
        reconsutructive_loss = self.mse_loss(y_pred_1, y_true_1) + self.mse_loss(y_pred_2, y_true_2)
        return contrastive_loss + (self.mse_weight * reconsutructive_loss)



def get_loss_function(scheme: str) -> nn.Module:
    '''
    Get the loss function for the given pretraining scheme.
    
    Parameters
    ----------
    scheme : str
        The pretraining scheme.
    
    Returns
    -------
    nn.Module
        The loss function.
    '''
    if scheme == 'hsv':
        return nn.MSELoss()
    if scheme == 'simclr':
        return NTXentLoss()
    if scheme == 'hsv_simclr':
        return HybridLoss()
    raise ValueError(f'Unknown pretraining scheme: {scheme}')


@cached
def cached_model_call(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    return model(X)


@cat_input_tensor
def cached_contrastive_loss_call(z_0: torch.Tensor, z_1: torch.Tensor) -> torch.Tensor:
    return nt_xent_loss(torch.cat([z_0, z_1], dim=0)).to(z_0.device)


class LARS(Optimizer):
    """Implements LARS (Layer-wise Adaptive Rate Scaling). Code adapted from: 
    https://github.com/4uiiurz1/pytorch-lars/blob/3d1f02dc86792e393552e054789f6b9349d2cc4e/lars.py#L4C1-L92C20

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)

    Example:
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, 
        params, 
        lr=required, 
        momentum=0, 
        eta=1e-3, 
        dampening=0,
        weight_decay=0, 
        nesterov=False, 
        epsilon=0
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm +
                        weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    # d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=(-local_lr * group['lr']))

        return loss



class ProfilerHistory:
    
    def __init__(self, device: torch.device):
        
        self.device = device
        self.profiler_history_dict = {
            'epoch': [],
            'phase': [],
            'time': [],
            'mem_usage': [],
            'mem_alloc': [],
            'mem_cache': [],
            'power_draw': [],
            'gpu_util': [],
            'temperature': [],
        }
    
    def update(self, epoch: int, phase: str, time: int) -> None:
        
        self.profiler_history_dict['epoch'].append(epoch)
        self.profiler_history_dict['phase'].append(phase)
        self.profiler_history_dict['time'].append(time)
        self.profiler_history_dict['mem_usage'].append(torch.cuda.memory_usage(self.device))
        self.profiler_history_dict['mem_alloc'].append(torch.cuda.memory_allocated(self.device))
        self.profiler_history_dict['mem_cache'].append(torch.cuda.memory_reserved(self.device))
        self.profiler_history_dict['power_draw'].append(torch.cuda.power_draw(self.device))
        self.profiler_history_dict['gpu_util'].append(torch.cuda.utilization(self.device))
        self.profiler_history_dict['temperature'].append(torch.cuda.temperature(self.device))
    
    def save(self, path: str) -> None:
        
        n_rows = len(self.profiler_history_dict['epoch'])
        df = pd.DataFrame(self.profiler_history_dict, index=list(range(n_rows)))
        df.to_csv(path, index=False)



def get_datetime(surrounding_brackets: bool=True) -> str:
    
    dt = datetime.datetime.now(datetime.timezone.utc)
    if surrounding_brackets:
        return '[' + dt.strftime('%Y-%m-%d %H:%M:%SZ') + ']'
    return dt.strftime('%Y-%m-%d %H:%M:%SZ')
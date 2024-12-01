import os
from typing import Tuple
from torch.optim.optimizer import Optimizer, required
from torch.nn import functional as F

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



def nt_xent_loss(x, temperature: float=0.5):
    """
    Normalized temperature-scaled cross entropy loss for self-supervised learning.
    Code adapted from: https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, D) where N is the batch size and D is the feature dimension.
    temperature : float, optional
        Temperature scaling factor, by default 0.5
        NOTE: the authors argue that the optimal temperature is 0.5 for most batch
            sizes, but tends to approach 0.1 as the batch size increases
    
    
    """
    assert len(x.size()) == 2
    
    # Cosine similarity
    xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    xcs[torch.eye(x.size(0)).bool()] = float("-inf")

    # Ground truth labels
    target = torch.arange(8)
    target[0::2] += 1
    target[1::2] -= 1
    
    # Standard cross entropy loss
    return F.cross_entropy(xcs / temperature, target, reduction="mean")



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

    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, nesterov=False, epsilon=0):
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
                    d_p.add_(weight_decay, p.data)
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

                p.data.add_(-local_lr * group['lr'], d_p)

        return loss
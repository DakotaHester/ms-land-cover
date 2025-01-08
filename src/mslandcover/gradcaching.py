'''
Utility functions for caching gradients during training. This process is 
necessary for contrastive learning approaches that require relatively large
batch sizes (i.e., SimCLR). By caching the forward pass of the model, the 
backward pass can be computed in smaller batches to reduce memory usage.

Code below is modified from the `GradCache` package, which can be found at
https://github.com/luyug/GradCache.

For more information, see the paper: https://arxiv.org/abs/2101.06983
'''

from functools import wraps
from typing import Callable, Dict, List, Union, Tuple, Any

import torch
from torch.utils.checkpoint import get_device_states, set_device_states
from torch import Tensor, distributed

from .utils import get_torch_device

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None



def cached(func: Callable[..., Tensor]):
    """
    A decorator that takes a model call function into a cached compatible version.
    :param func: A function that calls the model and return representation tensor.
    :return: A function that returns 1) representation leaf tensors for cache construction, 2) a closure function for
    the 2nd forward and the cached backward. Call 2) with 1) as argument after calling backward on the loss Tensor.
    """
    @wraps(func)
    def cache_func(*args, **kwargs):
        rnd_state = RandContext()
        with torch.no_grad():
            reps_no_grad = func(*args, **kwargs)
        if isinstance(reps_no_grad, Tensor):
            reps_no_grad = (reps_no_grad, )
        else:
            assert all(isinstance(v, Tensor) for v in reps_no_grad)
        leaf_reps = tuple(t.detach().requires_grad_() for t in reps_no_grad)
        
        # need to handle case that model returns two or more tensors for multi-task learning

        @wraps(func)
        def forward_backward_func(cache_reps: Union[Tensor, Tuple[Tensor]]):
            with rnd_state:
                reps = func(*args, **kwargs)
            if isinstance(reps, Tensor):
                reps = (reps,)
            if isinstance(cache_reps, Tensor):                
                cache_reps = (cache_reps,)
            
            # if the model returns multiple tensors, we need to find which
            # ones correspond to the cached representations passed to this function
            if len(reps) > 1:
                cr_size = cache_reps[0].shape[1:]
                for r in reps:
                    if r.size()[1:] == cr_size:
                        reps = (r,)
                        break
                if len(reps) > 1:
                    raise ValueError('Could not find the correct representation tensor in the model output.')
            
            surrogate = sum(map(lambda u, v: torch.dot(u.flatten(), v.grad.flatten()), reps, cache_reps), 0)
            
            try:
                surrogate.backward()
            except RuntimeError as e:
                print(f'RuntimeError: {e}')
                print("This is most likely caused by the gradients being too large for CUDA. Try reducing the batch size.")
                raise e

        return leaf_reps + (forward_backward_func,)
    return cache_func


def _cat_tensor_list(xx):
    if isinstance(xx, list) and len(xx) > 0 and all(isinstance(x, Tensor) for x in xx):
        return torch.cat(xx)
    else:
        return xx


def cat_input_tensor(func: Callable[..., Tensor]):
    """
    A decorator that concatenates positional and keyword arguments of type List[Tensor] into a single Tensor
    on the 0 dimension. This can come in handy dealing with results of representation tensors from multiple
    cached forward.
    :param func: A loss function
    :return: Decorated loss function for cached results.
    """
    @wraps(func)
    def cat_f(*args, **kwargs):
        args_cat = [_cat_tensor_list(x) for x in args]
        # for k, v in kwargs.items():
        #     print(f'k: {k}, v: {v}')
        kwargs_cat = dict((k, _cat_tensor_list(v)) for k, v in kwargs.values())
        return func(*args_cat, **kwargs_cat)
    return cat_f


def _maybe_gather_tensor(t: Any, axis: int):
    if not isinstance(t, Tensor):
        return t
    gathered = [torch.empty_like(t) for _ in range(distributed.get_world_size())]
    distributed.all_gather(gathered, t)
    gathered[distributed.get_rank()] = t
    return torch.cat(gathered, dim=axis)


def gather_input_tensor(func: Callable[..., Tensor], axis=0):
    """
    A decorator that all-gather positional and keyword arguments of type Tensor and concatenate them on axis.
    Intended to be used with distributed contrastive learning loss.
    :param func: A loss function
    :param axis: The axis the gathered tensors are concatenated.
    :return: Decorated loss function for distributed training.
    """
    @wraps(func)
    def f(*args, **kwargs):
        args_gathered = [_maybe_gather_tensor(x, axis=axis) for x in args]
        kwargs_gathered = dict((k, _maybe_gather_tensor(v, axis=axis)) for k, v in kwargs.values())
        return func(*args_gathered, **kwargs_gathered)
    return f



@cached
@autocast(get_torch_device().type)
def cached_model_call(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    return model(X)



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

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
from typing import Callable, Union, Tuple, Any

import torch
from torch.utils.checkpoint import get_device_states, set_device_states
from torch import Tensor, distributed


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

        @wraps(func)
        def forward_backward_func(cache_reps: Union[Tensor, Tuple[Tensor]]):
            with rnd_state:
                reps = func(*args, **kwargs)
            if isinstance(reps, Tensor):
                reps = (reps,)
            if isinstance(cache_reps, Tensor):
                cache_reps = (cache_reps,)
                            
            # NOTE: THE FOLLOWING IS A VERY VERY SPECIAL CASE. BE WARNED.
            # When the model returns a tuple of tensors, we assume that the first
            # Tensor corresponds to a reconstruction and the second Tensor corresponds
            # to the representation. We only cache the representation tensor.
            if len(reps) == 2:
                reps = (reps[1],)

            surrogate = sum(map(lambda u, v: torch.dot(u.flatten(), v.grad.flatten()), reps, cache_reps), 0)
            surrogate.backward()

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
        for k, v in kwargs.items():
            print(f'k: {k}, v: {v}')
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

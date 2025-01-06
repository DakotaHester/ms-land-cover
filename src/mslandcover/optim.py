"""
This module implements the PCGrad (Project Conflicting Gradient) gradient surgery
algorithm for multitask learning (https://arxiv.org/abs/2001.06782) and the 
Layer-wise Adaptive Rate Scaling (LARS) optimizer for training models with large 
batch sizes. 

"""
import copy
import random
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required



class PCGradAMP():
    """
    This class implements PCGrad with compatibility with PyTorch's AMP (Automatic 
    Mixed Precision) API. It also supports gradient accumulation for multitask
    objectives. This code is adapted from https://github.com/anzeyimana/Pytorch-PCGrad-GradVac-AMP-GradAccum/blob/master/pcgrad_amp.py
    To comply with the MIT License, the original license is included below:
    
    MIT License
    
    Copyright (c) 2022 Antoine Nzeyimana
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    # TODO: Include citation in manuscript    
    @misc{Pytorch-PCGrad-GradVac-AMP-GradAccum,
    author = {Antoine Nzeyimana},
    title = {Pytorch-PCGrad-GradVac-AMP-GradAccum/Antoine Nzeyimana},
    url = {https://github.com/anzeyimana/Pytorch-PCGrad-GradVac-AMP-GradAccum},
    year = {2022}
    }
    """
    def __init__(self, num_tasks, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler = None, reduction='sum', cpu_offload: bool = False):
        self.num_tasks = num_tasks
        self.cpu_offload = cpu_offload
        self._scaler, self._optim, self._reduction = scaler, optimizer, reduction
        # Setup default accumulated gradient
        self.accum_grad = []
        for i in range(self.num_tasks):
            grad, shape, has_grad = self._retrieve_grad()
            self.accum_grad.append((grad, shape, has_grad))
        return

    def state_dict(self) -> dict:
        if self._scaler is not None:
            return {'scaler': self._scaler.state_dict(), 'optimizer': self._optim.state_dict()}
        else:
            return {'optimizer': self._optim.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict['scaler'])
            self._optim.load_state_dict(state_dict['optimizer'])
        else:
            self._optim.load_state_dict(state_dict['optimizer'])

    @property
    def optimizer(self):
        return self._optim

    @property
    def scaler(self):
        return self._scaler

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        ret = self._optim.zero_grad()
        # Setup zero accumulated gradient
        for i in range(self.num_tasks):
            self.accum_grad[i][0].zero_()
            self.accum_grad[i][2].zero_()
        return ret

    def step(self):
        '''
        update the parameters with the gradient
        '''
        grads, shapes, has_grads = self._pack_accum_grads()
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

        if self._scaler is not None:
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            self._optim.step()

        return self.zero_grad()

    def backward(self, mt_losses):
        # Gradient accumulation
        for loss_id, loss in enumerate(mt_losses):
            self._optim.zero_grad()
            retain_graph = (loss_id < (self.num_tasks - 1))
            if self._scaler is not None:
                self._scaler.scale(loss).backward(retain_graph = retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)
            grad, shape, has_grad = self._retrieve_grad()
            acc_grad, acc_shape, acc_has_grad = self.accum_grad[loss_id]
            acc_grad += grad
            acc_has_grad = torch.logical_or(acc_has_grad, grad).to(dtype=acc_has_grad.dtype)
            self.accum_grad[loss_id] = (acc_grad, acc_shape, acc_has_grad)
        self._optim.zero_grad()

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx].to(p.device)
                idx += 1
        return

    def _pack_accum_grads(self):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for (grad, shape, has_grad) in self.accum_grad:
            grads.append(grad)
            has_grads.append(has_grad)
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    if self.cpu_offload:
                        grad.append(torch.zeros_like(p).cpu())
                        has_grad.append(torch.zeros_like(p, dtype=torch.int8).cpu())
                    else:
                        grad.append(torch.zeros_like(p).to(p.device))
                        has_grad.append(torch.zeros_like(p, dtype=torch.int8).to(p.device))
                else:
                    shape.append(p.grad.shape)
                    if self.cpu_offload:
                        grad.append(p.grad.detach().cpu())
                        has_grad.append(torch.ones_like(p, dtype=torch.int8).cpu())
                    else:
                        grad.append(p.grad.clone())
                        has_grad.append(torch.ones_like(p, dtype=torch.int8).to(p.device))
        grad_flatten = self._flatten_grad(grad, shape)
        has_grad_flatten = self._flatten_grad(has_grad, shape)
        return grad_flatten, shape, has_grad_flatten




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



class UncertainLossWeighter(torch.nn.Module):
    '''
    Wiegh loss by maximizing the gaussian likelihood of with homoscedastic uncertainty.
    Code adapted from https://github.com/anzeyimana/Pytorch-PCGrad-GradVac-AMP-GradAccum/blame/master/loss_weight.py
    Paper here: https://arxiv.org/pdf/1705.07115
    '''
    def __init__(self, num_tasks):
        super(UncertainLossWeighter, self).__init__()
        self.num_tasks = num_tasks
        self.log_vars = torch.nn.Parameter(torch.zeros((num_tasks)))

    def forward(self, mt_losses):
        assert len(mt_losses) == self.num_tasks
        weighted_losses = []
        for i,loss in enumerate(mt_losses):
            precision0 = torch.exp(-self.log_vars[i])
            wloss = precision0 * loss + self.log_vars[i]
            weighted_losses.append(wloss)
        return weighted_losses
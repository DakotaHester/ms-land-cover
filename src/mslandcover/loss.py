import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from .gradcaching import cat_input_tensor
from .utils import get_torch_device

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


@cat_input_tensor
@autocast(get_torch_device().type)
def cached_contrastive_loss_call(z_0: torch.Tensor, z_1: torch.Tensor) -> torch.Tensor:
    return nt_xent_loss(z_0, z_1).to(z_0.device)



@cat_input_tensor
@autocast(get_torch_device().type)
def cached_mse_loss_call(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true, reduction='sum')


class FocalLoss(torch.nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    Code pulled from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self, alpha=None, gamma=2., ignore_index=-100, reduction='sum'):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 2.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """

        super().__init__()
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha)
            alpha = alpha.float()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = torch.nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, x, y):
        
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0: return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        # print(x.dtype)
        log_p = F.log_softmax(x, dim=-1)
        y = y.long() # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/2
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss
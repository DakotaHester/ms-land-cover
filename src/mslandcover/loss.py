import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from .gradcaching import cat_input_tensor
from .utils import get_torch_device

from typing import Optional, Union, List, Literal

def nt_xent_loss(z_0: torch.Tensor, z_1: torch.Tensor, temperature: float=100.0, reduction='sum') -> torch.Tensor:
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


# class FocalLoss(torch.nn.Module):
#     """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
#     Code pulled from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

#     It is essentially an enhancement to cross entropy loss and is
#     useful for classification tasks when there is a large class imbalance.
#     x is expected to contain raw, unnormalized scores for each class.
#     y is expected to contain class labels.

#     Shape:
#         - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
#         - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
#     """

#     def __init__(self, alpha=None, gamma=2., ignore_index=-100, reduction='sum'):
#         """Constructor.

#         Args:
#             alpha (Tensor, optional): Weights for each class. Defaults to None.
#             gamma (float, optional): A constant, as described in the paper.
#                 Defaults to 2.
#             ignore_index (int, optional): class label to ignore.
#                 Defaults to -100.
#         """

#         super().__init__()
#         if alpha is not None:
#             if not isinstance(alpha, torch.Tensor):
#                 alpha = torch.tensor(alpha)
#             alpha = alpha.float()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#         self.nll_loss = torch.nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

#     def forward(self, y, x):
        
#         if x.ndim > 2:
#             # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
#             c = x.shape[1]
#             x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
#             # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
#             y = y.view(-1)
        
#         unignored_mask = y != self.ignore_index
#         y = y[unignored_mask]
#         if len(y) == 0: return torch.tensor(0.)
#         x = x[unignored_mask]

#         # compute weighted cross entropy term: -alpha * log(pt)
#         # (alpha is already part of self.nll_loss)
#         # print(x.dtype)
#         log_p = F.log_softmax(x, dim=-1)
#         y = y.long() # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/2
#         ce = self.nll_loss(log_p, y)

#         # get true class column from each row
#         all_rows = torch.arange(len(x))
#         log_pt = log_p[all_rows, y]

#         # compute focal term: (1 - pt)^gamma
#         pt = log_pt.exp()
#         focal_term = (1 - pt)**self.gamma

#         # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
#         loss = focal_term * ce
        
#         if self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'mean':
#             return loss.mean()
#         else:
#             return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for multi-class semantic segmentation.
    https://arxiv.org/abs/1810.07842

    Attributes
    ----------
    alpha : Optional[Union[float, List[float]]]
        Class weights for addressing class imbalance. If a list is provided, each value corresponds to a class weight.
    gamma : float
        Focusing parameter to control the degree of penalization for hard-to-classify regions.
    delta : float
        Tversky coefficient parameter to control the trade-off between false negatives and false positives.
    smooth : float
        Smoothing term to avoid division by zero.
    reduction : Literal['mean', 'sum', 'none']
        Specifies the reduction method to apply to the output. Options are 'mean', 'sum', or 'none'.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float]]] = None,
        gamma: float = 0.75,
        delta: float = 0.7,
        smooth: float = 1e-6,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        """
        Initialize the FocalTverskyLoss class.

        Parameters
        ----------
        alpha : Optional[Union[float, List[float]]]
            Class weights for addressing class imbalance.
        gamma : float
            Focusing parameter for controlling penalization of hard examples.
        delta : float
            Weighting parameter for false negatives and false positives in the Tversky index.
        smooth : float
            Smoothing factor to avoid division by zero.
        reduction : Literal['mean', 'sum', 'none']
            Specifies the reduction method for the loss output.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.smooth = smooth
        self.reduction = reduction
        
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Focal Tversky loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted probabilities with shape (batch_size, num_classes, ...).
        y_true : torch.Tensor
            Ground truth labels with shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Computed loss. The shape depends on the `reduction` parameter.
        """
        # Clamp predictions to prevent extreme values
        y_pred = torch.clamp(y_pred, self.smooth, 1.0 - self.smooth)
        
        # Convert labels to one-hot encoding
        num_classes = y_pred.shape[1]
        y_true = F.one_hot(y_true.long(), num_classes).permute(0, -1, *range(1, y_true.dim()))
        y_true = y_true.float()
        
        # Apply class weights if specified
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(y_pred.device)
            else:
                alpha = torch.tensor([self.alpha] * num_classes).to(y_pred.device)
            y_true = y_true * alpha.view(1, -1, *([1] * (y_true.dim() - 2)))
        
        # Calculate true positives, false negatives, and false positives
        dims = tuple(range(2, y_true.dim()))
        tp = torch.sum(y_true * y_pred, dims)
        fn = torch.sum(y_true * (1 - y_pred), dims)
        fp = torch.sum((1 - y_true) * y_pred, dims)
        
        # Calculate numerator and denominator separately for better stability
        numerator = tp + self.smooth
        denominator = tp + self.delta * fn + (1 - self.delta) * fp + self.smooth
        
        # Ensure denominator is not too close to zero
        denominator = torch.clamp(denominator, min=self.smooth)
        
        # Calculate Tversky index
        tversky = numerator / denominator
        
        # Clamp tversky index to prevent unstable power operation
        tversky = torch.clamp(tversky, self.smooth, 1.0 - self.smooth)
        
        # Apply focal term with safe power operation
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_tversky)
        elif self.reduction == 'sum':
            return torch.sum(focal_tversky)
        else:  # 'none'
            return focal_tversky

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class semantic segmentation.
    https://arxiv.org/abs/1708.02002

    Attributes
    ----------
    alpha : Optional[Union[float, List[float]]]
        Class weights for addressing class imbalance.
    gamma : float
        Focusing parameter to penalize hard examples.
    smooth : float
        Smoothing term to avoid instability during logarithmic operations.
    reduction : Literal['mean', 'sum', 'none']
        Specifies the reduction method for the loss output.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float]]] = None,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        """
        Initialize the FocalLoss class.

        Parameters
        ----------
        alpha : Optional[Union[float, List[float]]]
            Class weights for addressing class imbalance.
        gamma : float
            Focusing parameter for controlling penalization of hard examples.
        smooth : float
            Smoothing factor to avoid instability in logarithmic computations.
        reduction : Literal['mean', 'sum', 'none']
            Specifies the reduction method for the loss output.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Focal loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted probabilities with shape (batch_size, num_classes, ...).
        y_true : torch.Tensor
            Ground truth labels with shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Computed loss. The shape depends on the `reduction` parameter.
        """
        # Clamp predictions to prevent extreme values
        y_pred = torch.clamp(y_pred, self.smooth, 1.0 - self.smooth)
        
        # Convert labels to one-hot encoding
        num_classes = y_pred.shape[1]
        y_true = F.one_hot(y_true.long(), num_classes).permute(0, -1, *range(1, y_true.dim()))
        y_true = y_true.float()
        
        # Calculate focal loss with stable log
        log_prob = torch.log(y_pred)
        prob = torch.exp(log_prob)
        
        # Calculate focal term
        focal_term = torch.pow(1 - prob, self.gamma)
        
        # Combine terms
        focal_loss = -y_true * focal_term * log_prob
        
        # Apply class weights if specified
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(y_pred.device)
            else:
                alpha = torch.tensor([self.alpha] * num_classes).to(y_pred.device)
            focal_loss = alpha.view(1, -1, *([1] * (focal_loss.dim() - 2))) * focal_loss
        
        # Sum over spatial dimensions
        dims = tuple(range(2, y_true.dim()))
        focal_loss = torch.sum(focal_loss, dims)
        
        # Handle any remaining numerical instabilities
        focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:  # 'none'
            return focal_loss

class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss, combining Focal Tversky Loss and Focal Loss.
    https://www.sciencedirect.com/science/article/pii/S0895611121001750

    Attributes
    ----------
    weight : float
        Weight for the Focal Tversky loss component in the unified loss.
    reduction : Literal['mean', 'sum', 'none']
        Specifies the reduction method for the loss output.
    focal_tversky : FocalTverskyLoss
        Instance of the FocalTverskyLoss class.
    focal : FocalLoss
        Instance of the FocalLoss class.
    """
    
    def __init__(
        self,
        weight: float = 0.5,
        alpha: Optional[Union[float, List[float]]] = None,
        delta: float = 0.6,
        gamma: float = 0.5,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        """
        Initialize the UnifiedFocalLoss class.

        Parameters
        ----------
        weight : float
            Weight for the Focal Tversky loss in the unified loss.
        alpha : Optional[Union[float, List[float]]]
            Class weights for addressing class imbalance.
        delta : float
            Weighting parameter for false negatives and false positives in the Tversky index.
        gamma : float
            Focusing parameter for controlling penalization of hard examples.
        reduction : Literal['mean', 'sum', 'none']
            Specifies the reduction method for the loss output.
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        
        # Initialize component losses
        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha,
            gamma=gamma,
            delta=delta,
            reduction=reduction
        )
        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction
        )
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Unified Focal Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted probabilities with shape (batch_size, num_classes, ...).
        y_true : torch.Tensor
            Ground truth labels with shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Computed unified loss. The shape depends on the `reduction` parameter.
        """
        """Forward pass with improved numerical stability."""
        focal_tversky_loss = self.focal_tversky(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        
        # Combine losses with numerical stability check
        if self.reduction == 'none':
            combined_loss = self.weight * focal_tversky_loss + (1 - self.weight) * focal_loss
        else:
            combined_loss = (self.weight * focal_tversky_loss) + ((1 - self.weight) * focal_loss)
        
        # Final numerical stability check
        combined_loss = torch.nan_to_num(combined_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return combined_loss
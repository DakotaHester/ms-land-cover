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

class SegmentationLoss(nn.Module):
    """Base class for segmentation loss functions.
    
    Parameters
    ----------
    alpha : Optional[Union[float, List[float], torch.Tensor]]
        Class weights for handling class imbalance. Can be a float, list of floats, 
        or tensor of shape (C,) where C is the number of classes. If None, classes 
        are weighted equally.
    reduction : str
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """
    
    def __init__(
        self, 
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        
    def _handle_weights(self, num_classes: int) -> Optional[Tensor]:
        """Processes class weights into appropriate tensor format.
        
        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
            
        Returns
        -------
        Optional[Tensor]
            Processed weights tensor of shape (C,) or None if no weights specified.
        """
        if self.alpha is None:
            return None
            
        if isinstance(self.alpha, (float, int)):
            weights = torch.full((num_classes,), self.alpha)
        elif isinstance(self.alpha, list):
            weights = torch.tensor(self.alpha)
        else:
            weights = self.alpha
            
        return weights.float()
    
    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Applies reduction method to loss tensor.
        
        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor to be reduced.
            
        Returns
        -------
        torch.Tensor
            Reduced loss value.
        """
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # sum
            return loss.sum()


class DiceLoss(SegmentationLoss):
    """Dice loss for multi-class semantic segmentation.
    
    Parameters
    ----------
    alpha : Optional[Union[float, List[float], torch.Tensor]]
        Class weights for handling class imbalance.
    delta : float
        Controls weight given to false positive and false negatives.
        Must be in [0, 1].
    smooth : float
        Smoothing constant to prevent division by zero.
    reduction : str
        Reduction method to apply to the loss.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        delta: float = 0.5,
        smooth: float = 1e-6,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(alpha=alpha, reduction=reduction)
        self.delta = delta
        self.smooth = smooth
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted probabilities of shape (N, C, H, W)
        target : torch.Tensor
            Ground truth labels of shape (N, H, W) with values in [0, C-1]
            
        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        n_classes = input.shape[1]
        weights = self._handle_weights(n_classes)
        
        # Convert target to one-hot encoding
        target_onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
        
        # Calculate true positives, false negatives and false positives
        tp = torch.sum(target_onehot * input, dim=(2, 3))
        fn = torch.sum(target_onehot * (1 - input), dim=(2, 3))
        fp = torch.sum((1 - target_onehot) * input, dim=(2, 3))
        
        # Calculate Dice score for each class
        dice_score = (tp + self.smooth) / (tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        
        if weights is not None:
            dice_score = weights.to(dice_score.device) * dice_score
            
        # Calculate loss
        loss = 1 - dice_score
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        else:
            return self._reduce(loss)


class TverskyLoss(SegmentationLoss):
    """Tversky loss for multi-class semantic segmentation.
    
    Parameters
    ----------
    alpha : Optional[Union[float, List[float], torch.Tensor]]
        Class weights for handling class imbalance.
    delta : float
        Controls weight given to false positive and false negatives.
        Must be in [0, 1].
    smooth : float
        Smoothing constant to prevent division by zero.
    reduction : str
        Reduction method to apply to the loss.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        delta: float = 0.7,
        smooth: float = 1e-6,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(alpha=alpha, reduction=reduction)
        self.delta = delta
        self.smooth = smooth
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted probabilities of shape (N, C, H, W)
        target : torch.Tensor
            Ground truth labels of shape (N, H, W) with values in [0, C-1]
            
        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        n_classes = input.shape[1]
        weights = self._handle_weights(n_classes)
        
        # Convert target to one-hot encoding
        target_onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
        
        # Calculate true positives, false negatives and false positives
        tp = torch.sum(target_onehot * input, dim=(2, 3))
        fn = torch.sum(target_onehot * (1 - input), dim=(2, 3))
        fp = torch.sum((1 - target_onehot) * input, dim=(2, 3))
        
        # Calculate Tversky index for each class
        tversky_score = (tp + self.smooth) / (tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        
        if weights is not None:
            tversky_score = weights.to(tversky_score.device) * tversky_score
            
        # Calculate loss
        loss = 1 - tversky_score
        
        return self._reduce(loss)


class FocalLoss(SegmentationLoss):
    """Focal loss for multi-class semantic segmentation.
    
    Parameters
    ----------
    alpha : Optional[Union[float, List[float], torch.Tensor]]
        Class weights for handling class imbalance.
    gamma : float
        Focal parameter controls degree of down-weighting easy examples.
    reduction : str
        Reduction method to apply to the loss.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(alpha=alpha, reduction=reduction)
        self.gamma = gamma
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted probabilities of shape (N, C, H, W)
        target : torch.Tensor
            Ground truth labels of shape (N, H, W) with values in [0, C-1]
            
        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        n_classes = input.shape[1]
        weights = self._handle_weights(n_classes)
        
        # Convert target to one-hot encoding
        target_onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
        
        # Compute focal loss
        ce_loss = -target_onehot * torch.log(input + 1e-6)
        focal_weight = torch.pow(1 - input, self.gamma)
        focal_loss = focal_weight * ce_loss
        
        # Sum over spatial dimensions
        focal_loss = torch.sum(focal_loss, dim=(2, 3))
        
        if weights is not None:
            focal_loss = weights.to(focal_loss.device) * focal_loss
            
        return self._reduce(focal_loss)


class ComboLoss(SegmentationLoss):
    """Combination of Dice and Focal loss for multi-class semantic segmentation.
    
    Parameters
    ----------
    alpha : Optional[Union[float, List[float], torch.Tensor]]
        Class weights for handling class imbalance.
    weight : float
        Weight factor between Dice and Focal loss. Must be in [0, 1].
    gamma : float
        Focal parameter for the Focal loss component.
    delta : float
        Delta parameter for the Dice loss component.
    smooth : float
        Smoothing constant to prevent division by zero.
    reduction : str
        Reduction method to apply to the loss.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        weight: float = 0.5,
        gamma: float = 2.0,
        delta: float = 0.5,
        smooth: float = 1e-6,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(alpha=alpha, reduction=reduction)
        self.weight = weight
        self.dice_loss = DiceLoss(alpha=alpha, delta=delta, smooth=smooth, reduction='none')
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted probabilities of shape (N, C, H, W)
        target : torch.Tensor
            Ground truth labels of shape (N, H, W) with values in [0, C-1]
            
        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        dice_loss = self.dice_loss(input, target)
        focal_loss = self.focal_loss(input, target)
        
        combo_loss = self.weight * dice_loss + (1 - self.weight) * focal_loss
        
        return self._reduce(combo_loss)


class FocalTverskyLoss(SegmentationLoss):
    """Focal Tversky loss for multi-class semantic segmentation.
    
    Parameters
    ----------
    alpha : Optional[Union[float, List[float], torch.Tensor]]
        Class weights for handling class imbalance.
    delta : float
        Controls weight given to false positive and false negatives.
    gamma : float
        Focal parameter controls degree of down-weighting easy examples.
    smooth : float
        Smoothing constant to prevent division by zero.
    reduction : str
        Reduction method to apply to the loss.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        super().__init__(alpha=alpha, reduction=reduction)
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted probabilities of shape (N, C, H, W)
        target : torch.Tensor
            Ground truth labels of shape (N, H, W) with values in [0, C-1]
            
        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        n_classes = input.shape[1]
        weights = self._handle_weights(n_classes)
        
        # Convert target to one-hot encoding
        target_onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
        
        # Calculate true positives, false negatives and false positives
        tp = torch.sum(target_onehot * input, dim=(2, 3))
        fn = torch.sum(target_onehot * (1 - input), dim=(2, 3))
        fp = torch.sum((1 - target_onehot) * input, dim=(2, 3))
        
        # Calculate Tversky index for each class
        tversky_score = (tp + self.smooth) / (tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        
        # Apply focal weighting
        focal_tversky_loss = torch.pow(1 - tversky_score, self.gamma)
        
        if weights is not None:
            focal_tversky_loss = weights.to(focal_tversky_loss.device) * focal_tversky_loss
            
        return self._reduce(focal_tversky_loss)
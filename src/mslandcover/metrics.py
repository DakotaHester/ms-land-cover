import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np

def tensor_to_numpy(tensor: torch.tensor, type='int', flatten=True) -> np.ndarray:
    
    if tensor.device != 'cpu':
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor = tensor.cpu()
    arr = tensor.numpy().astype(type)
    if flatten:
        arr = arr.flatten()
    return arr

def numpyify(func, type='int', flatten=True):
    

    def wrapper(*args, **kwargs):
        args = [tensor_to_numpy(arg, type=type, flatten=flatten) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: tensor_to_numpy(v, type=type, flatten=flatten) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    
    # wrapper function should inherit all attributes of the original function
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    wrapper.__annotations__ = func.__annotations__
    wrapper.__dict__.update(func.__dict__)
    
    return wrapper


@numpyify
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy of the model.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The accuracy of the model.
    """
    return metrics.accuracy_score(y_true, y_pred, )


@numpyify
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the F1 score of the model.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The F1 score of the model.
    """
    return metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)



@numpyify
def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the precision score of the model (or user's accuracy).

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The precision score of the model.
    """
    return metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)



@numpyify
def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the recall score of the model (or producer's accuracy).

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The recall score of the model.
    """
    return metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)



@numpyify
def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the macro F1 score of the model.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The macro F1 score of the model.
    """
    return metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)


@numpyify
def macro_precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the macro precision score of the model. (or user's accuracy)

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The macro precision score of the model.
    """
    return metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)



@numpyify
def macro_recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    """
    Compute the macro recall score of the model. (or producer's accuracy)

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The macro recall score of the model.
    """
    return metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)


@numpyify
def kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Cohen's kappa score of the model.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The Cohen's kappa score of the model.
    """
    return metrics.cohen_kappa_score(y_true, y_pred)



def psnr(img1: torch.Tensor, img2: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two tensors.
    Assumes input tensors are normalized to have mean 0 and std deviation 1.

    Args:
        img1 (torch.Tensor): First input image tensor.
        img2 (torch.Tensor): Second input image tensor.
        reduction (str): Reduction method: 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: PSNR value.
    """
    mse = F.mse_loss(img1, img2, reduction='none')
    psnr_value = 10 * torch.log10(1 / (mse + 1e-8))  # Small epsilon to prevent log(0)
    psnr_value = psnr_value.mean(dim=[1, 2, 3])  # Average over channel and spatial dims
    
    if reduction == 'mean':
        return psnr_value.mean()
    elif reduction == 'sum':
        return psnr_value.sum()
    else:
        return psnr_value



def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute the Structural Similarity Index Measure (SSIM) between two tensors.
    Assumes input tensors are normalized to have mean 0 and std deviation 1.

    Args:
        img1 (torch.Tensor): First input image tensor.
        img2 (torch.Tensor): Second input image tensor.
        window_size (int): Kernel size for local statistics.
        reduction (str): Reduction method: 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: SSIM value.
    """
    n_channels = img1.shape[1]
    C1 = 0.01 ** 2  # Stability constants (default values in SSIM paper)
    C2 = 0.03 ** 2
    
    # Compute local mean and variance
    pad = window_size // 2
    weight = torch.ones(n_channels, 1, window_size, window_size, device=img1.device) / (window_size ** 2)
    
    mu1 = F.conv2d(img1, weight, padding=pad, groups=n_channels)
    mu2 = F.conv2d(img2, weight, padding=pad, groups=n_channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, weight, padding=pad, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, weight, padding=pad, groups=n_channels) - mu2_sq
    sigma12 = F.conv2d((img1 * img2), weight, padding=pad, groups=n_channels) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map.mean(dim=(1, 2, 3))  # Average over channel and spatial dims
    
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'sum':
        return ssim_map.sum()
    else:
        return ssim_map
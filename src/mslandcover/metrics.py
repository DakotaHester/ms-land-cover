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
    return metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)



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
    return metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)



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
    return metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)



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
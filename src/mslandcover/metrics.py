import torch
import torch.nn.functional as F
from sklearn import metrics

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the accuracy of the model.

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The accuracy of the model.
    """
    return metrics.accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the F1 score of the model.

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The F1 score of the model.
    """
    return metrics.f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')



def precision_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the precision score of the model (or user's accuracy).

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The precision score of the model.
    """
    return metrics.precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')



def recall_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the recall score of the model (or producer's accuracy).

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The recall score of the model.
    """
    return metrics.recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')



def macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the macro F1 score of the model.

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The macro F1 score of the model.
    """
    return metrics.f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')


def macro_precision_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the macro precision score of the model. (or user's accuracy)

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The macro precision score of the model.
    """
    return metrics.precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')



def macro_recall_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    
    """
    Compute the macro recall score of the model. (or producer's accuracy)

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The macro recall score of the model.
    """
    return metrics.recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')


def kappa_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the Cohen's kappa score of the model.

    Parameters
    ----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns
    -------
    float
        The Cohen's kappa score of the model.
    """
    return metrics.cohen_kappa_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms

from typing import Optional, Union, Tuple

def random_resize_crop(
    X: torch.Tensor, 
    y: Optional[torch.Tensor]=None, 
    size: int=256, 
    seed: Optional[int]=None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Randomly crop the input image tensor. If a target tensor is provided, 
    crop it in the same way. This implementation is similar to torchvision.transforms.RandomResizedCrop,
    exceot that two tensors can be passed in and cropped in the same way.
    
    Parameters
    ----------
    X : torch.Tensor
        Input image tensor.
    y : torch.Tensor, optional
        Target tensor, by default None.
    size : int, optional
        Size of the crop, by default 256.
    seed : int, optional
        Random seed for reproducibility, by default None.
    
    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Cropped image tensor or a tuple of cropped image and target tensors.
    """
    
    if seed is not None:
        torch.manual_seed(seed)
    
    i, j, h, w = transforms.RandomCrop.get_params(X, output_size=(size, size))
    
    if y is not None:
        return F.resized_crop(X, i, j, h, w, size=size), F.resized_crop(y, i, j, h, w, size=size)
    return F.resized_crop(X, i, j, h, w, size=size)



def get_color_transforms(s: float=1.0) -> transforms.Compose:
    """
    Create a composition of color space augmentations.
    
    Parameters
    ----------
    s : float, optional
        Strength of the color augmentations, by default 1.0.
    
    Returns
    -------
    transforms.Compose
        Composition of color space augmentations.
    """
    return transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5)
    ])
    


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image tensor to an HSV image tensor. Code sourced from:
    https://github.com/limacv/RGB_HSV_HSL/blob/6dbbd0af542bc8a4000dffa14b8629b2e093bcdf/color_torch.py#L28
    
    Parameters
    ----------
    rgb : torch.Tensor
        Input RGB image tensor.
    
    Returns
    -------
    torch.Tensor
        Output HSV image tensor.
    """
    
    if len(rgb.size()) == 3:
        rgb = rgb.unsqueeze(0)
        
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)



def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert an HSV image tensor to an RGB image tensor. Code sourced from:
    https://github.com/limacv/RGB_HSV_HSL/blob/6dbbd0af542bc8a4000dffa14b8629b2e093bcdf/color_torch.py#L44
    
    Parameters
    ----------
    hsv : torch.Tensor
        Input HSV image tensor.
    
    Returns
    -------
    torch.Tensor
        Output RGB image tensor.
    """
    
    if len(hsv.size()) == 3:
        hsv = hsv.unsqueeze(0)
    
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    
    return rgb



class SimCLRDataAugmentation:
    """
    Data Transformer for creating a pair of views from an image.
    """
    
    def __init__(self):
        
        self.color_transforms = get_color_transforms()
    
    
    def __call__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Apply data augmentation to the input image tensor.
        
        Parameters
        ----------
        X : torch.Tensor
            Input image tensor.
        y : torch.Tensor, optional
            Target tensor, by default None.
        
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Augmented image tensor or a tuple of augmented image and target tensors.
        """
                
        # resize and random crop
        if y is not None:
            X_1, y_1 = random_resize_crop(X, y, size=256)
            X_2, y_2 = random_resize_crop(X, y, size=256)
        else:
            X_1 = random_resize_crop(X, size=256)
            X_2 = random_resize_crop(X, size=256)
        
        # color distortions
        X_1 = self.color_transforms(X_1)
        X_2 = self.color_transforms(X_2)
        
        if y is not None:
            return (X_1, y_1), (X_2, y_2)
        return X_1, X_2


class StandardDataAugmentations:
    '''
    Simple data augmentations for training a segmentation model. Includes random 
    flips and color distortions. No rotation or scaling is applied for simplicity
    '''
    
    def __init__(self):
        self.color_transforms = get_color_transforms()
    
    def __call__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        
        # do not resize this time, just apply random flip and color distortions
        if torch.rand(1) > 0.5:
            X = F.hflip(X)
            if y is not None:
                y = F.hflip(y)
        
        if torch.rand(1) > 0.5:
            X = F.vflip(X)
            if y is not None:
                y = F.vflip(y)
        
        X = self.color_transforms(X)
        
        if y is not None:
            return X, y
        return X
    
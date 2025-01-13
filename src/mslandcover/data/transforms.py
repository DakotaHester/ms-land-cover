import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
import numpy as np
from typing import Callable, List, Optional, Union, Tuple
import cv2 as cv
import os


class Clamp(Callable):
    def __init__(self, min_val: float=0.0, max_val: float=1.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor, min=self.min_val, max=self.max_val)


def random_resize_crop(
    X: torch.Tensor, 
    aspect_ratio_range: Tuple[float, float]=(3/4, 4/3),
    area_range: Tuple[float, float]=(0.08, 1.0),
    hflip_prob: float=0.5,
    size: int=256, 
    seed: Optional[int]=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Randomly crop the input image tensor. This implementation is similar to
    torchvision.transforms.RandomResizedCrop,
    exceot that two tensors can be passed in and cropped in the same way.
    
    Parameters
    ----------
    X : torch.Tensor
        Input image tensor.
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
    
    i, j, h, w = transforms.RandomResizedCrop.get_params(X, scale=area_range, ratio=aspect_ratio_range)
    
    X = F.resized_crop(X, i, j, h, w, size=(size, size))
    if y is not None:
        y = F.resized_crop(y, i, j, h, w, size=(size, size))
    
    if torch.rand(1) < hflip_prob:
        X = F.hflip(X)
        if y is not None:
            y = F.hflip(y)
    
    if y is not None:
        return X, y
    return X


def normalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    
    permuted = False
    if tensor.size(0) == 3:
        tensor = tensor.permute(1, 2, 0)
        permuted = True
    
    tensor = (tensor - mean) / std
    
    if permuted:
        tensor = tensor.permute(2, 0, 1)
    
    return tensor
    


def get_color_transforms(s: float=0.5) -> transforms.Compose:
    """
    Create a composition of color space augmentations.
    
    NOTE: In original implementation, s=1.0. For this implementation, the 
    strength of the color augmentations can be reduced due to the nature of the 
    imagery data.
    
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
            transforms.GaussianBlur(kernel_size=25, sigma=(0.1, 2.0)) # kernel size should be 10% of image size
        ], p=0.5),
        Clamp(),
    ])


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
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
        
    cmax, cmax_idx = torch.max(rgb, dim=0, keepdim=True)
    cmin = torch.min(rgb, dim=0, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[0:1, :, :])
    cmax_idx[delta == 0] = 3
        
    hsv_h[cmax_idx == 0] = (((rgb[1:2] - rgb[2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[2:3] - rgb[0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[0:1] - rgb[1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    
    hsv = torch.cat([hsv_h, hsv_s, hsv_v], dim=0)
    
    return hsv


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
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
    rank = len(hsv.size())
    if rank == 4:
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
    
    if rank == 3:
        return rgb.squeeze(0)
    return rgb



def add_gaussian_noise(tensor: torch.Tensor, mean: float=0.0, std: float=0.1) -> torch.Tensor:
    return tensor + (torch.randn_like(tensor) * std + mean).to(tensor.device)



def add_poisson_noise(tensor: torch.Tensor, lam: float=0.1) -> torch.Tensor:
    return tensor + torch.poisson(torch.ones_like(tensor) * lam).to(tensor.device)



def add_noise(tensor: torch.Tensor, std: float=0.1, lam: float=0.1) -> torch.Tensor:
    return add_gaussian_noise(tensor, std=std) + add_poisson_noise(tensor, lam=lam) - tensor



class SimCLRDataAugmentation:
    """
    Data Transformer for creating a pair of views from an image.
    """
    
    def __init__(self, size: int=96):
        
        scale_factor = (size ** 2) / (256 ** 2)
        scale_min = 0.08 * scale_factor
        scale_max = 1.0 * scale_factor
        
        self.size = size
        self.random_resize_crop = transforms.RandomResizedCrop(size=size, scale=(scale_min, scale_max))
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.color_transforms = get_color_transforms()
        self.composed_transforms = transforms.Compose([
            self.random_resize_crop,
            self.random_horizontal_flip,
            self.color_transforms,
        ])
    
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to the input image tensor.
        
        Parameters
        ----------
        X : torch.Tensor
            Input image tensor.
        
        Returns
        -------
        torch.Tensor
            Augmented image tensor 
        """
                
        # resize and random crop
        return self.composed_transforms(X)


class StandardDataAugmentations:
    '''
    Simple data augmentations for training a segmentation model. Includes random 
    flips and color distortions.
    '''
    
    def __init__(self, size: int=256):
        self.color_transforms = get_color_transforms()
        self.size = size
        self.resize_transform = ResizeTransform(size=size)
    
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
        
        rot_angle = torch.randint(0, 4, (1,)).item()
        X = F.rotate(X, rot_angle * 90)
        if y is not None:
            y = F.rotate(y, rot_angle * 90)
        
        X = self.color_transforms(X)
        
        if y is not None:
            X, y = self.resize_transform(X, y)
        else:
            X = self.resize_transform(X)
        
        if y is not None:
            return X, y
        return X


def visualize_transforms(
    out_dir: str,
    n_views: int, 
    return_hsv: bool,
    noisy_input: bool,
    dataset: torch.utils.data.Dataset, 
    data_paths: List[str], 
    pretrain_schema: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    n_examples: int=20,
) -> None:
    """
    Very crude function to load a few images from the dataset, visualize the
    augmentations, and save them to disk. Probably not the best way to do this,
    but it serves the purpose for now.
    """
    for i in range(n_examples):
        im_index = np.random.randint(len(dataset))
        og_img = cv.imread(data_paths[im_index])
        out_path = os.path.join(out_dir, pretrain_schema, str(im_index))
        os.makedirs(out_path, exist_ok=True)
        cv.imwrite(os.path.join(out_path, f'{i}_original.png'), og_img)
        if return_hsv:
            if n_views == 1:
                X, hsv = dataset[im_index]
                X = X.permute(1, 2, 0)
                X = ((X * std) + mean) * 255 # undo normalization
                X = X.int().cpu().numpy().astype(np.uint8)
                X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(out_path, f'{i}_view_0.png'), X)
                
                hsv = hsv.permute(1, 2, 0).cpu().numpy()
                hsv = (hsv * 255).astype(np.uint8)
                hsv = cv.cvtColor(hsv, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(out_path, f'{i}_view_0_hsv.png'), hsv)
            
            else:
                for view in range(n_views):
                    X, hsv = dataset[im_index][view]
                    X = X.permute(1, 2, 0)
                    X = ((X * std) + mean) * 255
                    X = X.int().cpu().numpy().astype(np.uint8)
                    X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
                    cv.imwrite(os.path.join(out_path, f'{i}_view_{view}.png'), X)

                    hsv = hsv.permute(1, 2, 0).cpu().numpy()
                    hsv = (hsv * 255).astype(np.uint8)
                    hsv = cv.cvtColor(hsv, cv.COLOR_RGB2BGR)
                    cv.imwrite(os.path.join(out_path, f'{i}_view_{view}_hsv.png'), hsv)
        
        elif noisy_input:
            if n_views == 1:
                noisy_X, X = dataset[im_index]
                X = X.permute(1, 2, 0)
                X = ((X * std) + mean) * 255
                X = X.int().cpu().numpy().astype(np.uint8)
                X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(out_path, f'{i}_view_0.png'), X)
                
                noisy_X = noisy_X.permute(1, 2, 0)
                noisy_X = ((noisy_X * std) + mean) * 255
                noisy_X = torch.clamp(noisy_X, min=0, max=255)
                noisy_X = noisy_X.int().cpu().numpy().astype(np.uint8)
                noisy_X = cv.cvtColor(noisy_X, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(out_path, f'{i}_view_0_noisy.png'), noisy_X)
            
            else:
                for view in range(n_views):
                    noisy_X, X = dataset[im_index][view]
                    X = X.permute(1, 2, 0)
                    X = ((X * std) + mean) * 255
                    X = X.int().cpu().numpy().astype(np.uint8)
                    X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
                    cv.imwrite(os.path.join(out_path, f'{i}_view_{view}.png'), X)

                    noisy_X = noisy_X.permute(1, 2, 0)
                    noisy_X = ((noisy_X * std) + mean) * 255
                    noisy_X = torch.clamp(noisy_X, min=0, max=255)
                    noisy_X = noisy_X.int().cpu().numpy().astype(np.uint8)
                    noisy_X = cv.cvtColor(noisy_X, cv.COLOR_RGB2BGR)
                    cv.imwrite(os.path.join(out_path, f'{i}_view_{view}_noisy.png'), noisy_X)
                
        else:
            if n_views == 1:
                X= dataset[im_index]
                X = X.permute(1, 2, 0)
                X = ((X * std) + mean) * 255 # undo normalization
                X = X.int().cpu().numpy().astype(np.uint8)
                X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(out_path, f'{i}_view_0.png'), X)
            
            else:
                for view in range(n_views):
                    X = dataset[im_index][view]
                    X = X.permute(1, 2, 0)
                    X = ((X * std) + mean) * 255
                    X = X.int().cpu().numpy().astype(np.uint8)
                    X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
                    cv.imwrite(os.path.join(out_path, f'{i}_view_{view}.png'), X)



class ResizeTransform:
    """
    Resize the input image tensor to the desired size.
    """
    
    def __init__(self, size: int=256):
        self.size = size
    
    def __call__(self, X: torch.Tensor, y: Optional[torch.Tensor]=None) -> torch.Tensor:
            
        if X.shape[1] > self.size:
            # crop a random region from the image to the desired size
            x_offset = torch.randint(0, X.shape[1] - self.size, (1,)).item()
            y_offset = torch.randint(0, X.shape[2] - self.size, (1,)).item()
            X = F.crop(X, x_offset, y_offset, self.size, self.size)
            
            if y is not None:
                y = F.crop(y, x_offset, y_offset, self.size, self.size)
            
        elif X.shape[1] < self.size:
            # resize to the desired size
            X = F.resize(X, (self.size, self.size))
            
            if y is not None:
                y = F.resize(y, (self.size, self.size))
        else:
            pass
            
        if y is not None:
            return X, y
        return X
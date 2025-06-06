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
    
    dim = tensor.dim()
    if dim == 3:
        tensor = (tensor - mean[:, None, None]) / std[:, None, None]
    
    elif dim == 4:
        tensor = (tensor - mean[None, :, None, None]) / std[None, :, None, None]
    
    return tensor
    


def get_color_transforms(s: float=0.5, kernel_size: int=25, scale_sigma_by_s: bool=False) -> transforms.Compose:
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

    # kernel size should be 10% of image size, odd, and greater than 1
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        kernel_size = 3
    
    if scale_sigma_by_s:
        sigma = (0.1*s, 2.0*s)
    else:
        sigma = (0.1, 2.0)
    
    return transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma) # kernel size should be 10% of image size
        ], p=0.5),
        Clamp(),
    ])
    



class RandomPerBandJitter:
    """
    Apply random brightness and contrast jitter independently per band.
    """
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        # img: Tensor of shape (C, H, W)
        C = img.shape[0]
        out = torch.empty_like(img)
        for c in range(C):
            band = img[c]
            b_factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
            c_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = band.mean()
            jittered = (band - mean) * c_factor + mean  # contrast
            jittered = jittered * b_factor              # brightness
            out[c] = jittered
        return out


class RandomGamma:
    """
    Apply gamma correction randomly to each band.
    """
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        out = torch.empty_like(img)
        for c in range(img.shape[0]):
            # make it so that gamma in (0.5, 1) is equally as likely as gamma in (1, 2)
            if np.random.rand() < 0.5:
                gamma = np.random.uniform(1.0, self.gamma_range[1])
            else:
                gamma = np.random.uniform(self.gamma_range[0], 1.0)
            # gamma = np.random.uniform(*self.gamma_range)
            out[c] = img[c] ** gamma
        return torch.clamp(out, 0, 1)


def get_multispectral_augmentations(s=1.0):
    """
    Returns a composition of augmentations suitable for 4-band multispectral data.
    """
    return transforms.Compose([
        RandomGamma(gamma_range=(0.5, 2.0)),
        RandomPerBandJitter(brightness=0.3*s, contrast=0.3*s),
        # transforms.RandomGrayscale(p=0.2),
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


def rgb_to_lab(rgb: torch.Tensor, contrast_enhance_factor=0) -> torch.Tensor:
    """
    Convert an RGB image tensor to a LAB image tensor.
    The output values are normalized to be between 0 and 1.
    
    Parameters
    ----------
    rgb : torch.Tensor
        Input RGB image tensor with values between 0 and 1
        Shape should be (3, H, W)
    
    Returns
    -------
    torch.Tensor
        Output LAB image tensor with values between 0 and 1
        Shape will be (3, H, W)
    """
    # RGB to XYZ conversion matrix
    rgb = torch.clamp(rgb, min=0, max=1)
    rgb_to_xyz_matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]).type_as(rgb)

    # Reference white point (D65)
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883]).type_as(rgb)

    # Helper function for the LAB conversion
    def f(x):
        delta = 6/29
        mask = x > delta**3
        result = torch.empty_like(x)
        result[mask] = torch.pow(x[mask], 1/3)
        result[~mask] = x[~mask]/(3 * delta**2) + 4/29
        return result

    # Convert RGB to XYZ
    rgb_linear = torch.where(rgb > 0.04045,
                           torch.pow((rgb + 0.055) / 1.055, 2.4),
                           rgb / 12.92)
    
    xyz = torch.einsum('ab,b...->a...', rgb_to_xyz_matrix, rgb_linear)
    
    # Convert XYZ to LAB
    xyz_normalized = xyz / xyz_ref_white.view(3, 1, 1)
    
    fx = f(xyz_normalized[0:1])
    fy = f(xyz_normalized[1:2])
    fz = f(xyz_normalized[2:3])
    
    # Calculate LAB values
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    if contrast_enhance_factor > 0:
        L = torch.clamp(L, min=contrast_enhance_factor, max=100-contrast_enhance_factor)
        # scale back to 0-1 range
        L = (L - contrast_enhance_factor) # (0, 100 - 2*contrast_enhance_factor)
        L = L / (100 - 2*contrast_enhance_factor) # (0, 1)
    else:
        L = L / 100

    # L = L / 100  # L goes from 0 to 100
    a = (a + 128) / 255  # a goes from -128 to +127
    b = (b + 128) / 255  # b goes from -128 to +127
    
    lab = torch.cat([L, a, b], dim=0)
    
    return lab


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



# def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    
#     return F.rgb_to_lab(rgb)


def gaussian_noise(tensor: torch.Tensor, mean: float=0.0, std: float=0.1) -> torch.Tensor:
    return (torch.randn_like(tensor) * std + mean).to(tensor.device)


def poisson_noise(tensor: torch.Tensor, lam: float=0.1) -> torch.Tensor:
    return torch.poisson(torch.ones_like(tensor) * lam).to(tensor.device)



def add_gaussian_noise(tensor: torch.Tensor, mean: float=0.0, std: float=0.1) -> torch.Tensor:
    return tensor + (torch.randn_like(tensor) * std + mean).to(tensor.device)



def add_poisson_noise(tensor: torch.Tensor, lam: float=0.1) -> torch.Tensor:
    return tensor + torch.poisson(torch.ones_like(tensor) * lam).to(tensor.device)


def add_noise(tensor: torch.Tensor, std: float=0.1, lam: float=0.1) -> torch.Tensor:
    return tensor + gaussian_noise(tensor, std=std) + poisson_noise(tensor, lam=lam) - poisson_noise(tensor, lam=lam)

# def add_noise(tensor: torch.Tensor, std: float=0.1, lam: float=0.1) -> torch.Tensor:
    # return add_gaussian_noise(tensor, std=std) + add_poisson_noise(tensor, lam=lam) - tensor


class Random90DegreeRotation:
    
    def __init__(self):
        pass
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Randomly rotate the input image tensor by 0, 90, 180, or 270 degrees.
        
        Parameters
        ----------
        X : torch.Tensor
            Input image tensor.
        
        Returns
        -------
        torch.Tensor
            Rotated image tensor.
        """
        
        rot_angle = torch.randint(0, 4, (1,)).item()
        return F.rotate(X, rot_angle * 90)



class SimCLRDataAugmentation:
    """
    Data Transformer for creating a pair of views from an image.
    """
    
    def __init__(self, size: int=192, s: float=1.0):
        
        self.size = size
        kernel_size = int(size*0.1)
        # self.resize_transform = ResizeTransform(size=size)
        self.random_resize_crop = transforms.RandomResizedCrop(size=size)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.color_transforms = get_color_transforms(s=s, kernel_size=kernel_size)
        self.composed_transforms = transforms.Compose([
            # self.resize_transform,
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




class SimpleRandomCrop:
    
    def __init__(self, size: int=128):
        self.size = size
    
    def __call__(self, X: torch.Tensor, y: Optional[torch.Tensor]=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # handle case where image is already the desired size
        if X.shape[1] == self.size and X.shape[2] == self.size:
            if y is not None:
                return X, y
            return X
        
        # determine upper left corner of crop - make sure that crop is completely within the image
        x_offset = torch.randint(0, X.shape[1] - self.size, (1,)).item()
        y_offset = torch.randint(0, X.shape[2] - self.size, (1,)).item()
        
        X = F.crop(X, x_offset, y_offset, self.size, self.size)
        if y is not None:
            y = F.crop(y, x_offset, y_offset, self.size, self.size)
            return X, y
        return X


class HiResDataAugmentation:
    """
    Instead of random resize and cropping, simply clip a random region from the 
    image at the desired size - no resizing, stretching, or squishing.
    Also, add VerticalFlip and random 90 degree rotation.
    """
    
    def __init__(self, size: int=192, s: float=1.0):
        
        self.size = size
        # self.resize_transform = ResizeTransform(size=size)
        # self.random_resize_crop = transforms.RandomResizedCrop(size=size)
        self.random_crop = SimpleRandomCrop(size=size)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.random_vertical_flip = transforms.RandomVerticalFlip()
        self.random_90_degree_rotation = Random90DegreeRotation()
        # self.elastic_transform = transforms.ElasticTransform(alpha=((size/256)*50.0)*5*s, sigma=((size/256)*5.0)*5*s)
        # self.color_transforms = get_color_transforms(s=s, kernel_size=int(size*0.1), scale_sigma_by_s=True)
        self.multispectral_augmentations = get_multispectral_augmentations()
        # self.gaussian_noise = transforms.Lambda(lambda x: add_gaussian_noise(x, std=0.1*s)) # gaussian noise handled by Dataset
        self.blur = transforms.GaussianBlur(kernel_size=0.1*size, sigma=(0.1*s, 2.0*s))
        self.composed_transforms = transforms.Compose([
            self.random_crop,
            self.random_horizontal_flip,
            self.random_vertical_flip,
            self.random_90_degree_rotation,
            self.multispectral_augmentations,
            # self.gaussian_noise,
            self.blur,
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
    
    def __init__(self, size: int=256, s: float=1.0, use_color_transforms: bool=True):
        
        self.color_transforms = None
        if use_color_transforms:
            self.color_transforms = get_color_transforms(s=s)
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
        
        if self.color_transforms is not None:
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
    return_lab: bool,
    return_si: bool,
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
        
        if return_si:
            X, si = dataset[im_index]
            X = X.permute(1, 2, 0)
            X = ((X * std) + mean) * 255 # undo normalization
            X = torch.clamp(X, min=0, max=255)
            X = X.int().cpu().numpy().astype(np.uint8)
            X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(out_path, f'{i}_view_0_og.png'), X)
            
            si = si.permute(1, 2, 0).cpu().numpy()
            si = si + 1
            si = (si * (255 / 2)).astype(np.uint8)
            si = cv.cvtColor(si, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(out_path, f'{i}_view_0_si.png'), si)
        
        else:
            for view in range(n_views):
                X, si = dataset[im_index][view]
                X = X.permute(1, 2, 0)
                X = ((X * std) + mean) * 255
                X = torch.clamp(X, min=0, max=255)
                X = X.int().cpu().numpy().astype(np.uint8)
                X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(out_path, f'{i}_view_{view}.png'), X)

                si = si.permute(1, 2, 0).cpu().numpy()
                si = si + 1
                si = (si * (255 / 2)).astype(np.uint8)
                si = cv.cvtColor(si, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(out_path, f'{i}_view_{view}_si.png'), si)
            
        
        if return_hsv or return_lab:
            if n_views == 1:
                X, hsv = dataset[im_index]
                X = X.permute(1, 2, 0)
                X = ((X * std) + mean) * 255 # undo normalization
                X = torch.clamp(X, min=0, max=255)
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
                    X = torch.clamp(X, min=0, max=255)
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
                # print(X.shape)
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



def calculate_ndvi(X: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from an input
    image tensor. The NDVI is calculated as follows:
    
    NDVI = (NIR - RED) / (NIR + RED)
    
    Parameters
    ----------
    X : torch.Tensor
        Input image tensor with shape (B, C, H, W) or (C, H, W) where C is the
        number of channels. Channels should be ordered as (NIR, R, G).
    eps : float, optional
        Small value to prevent division by zero, by default 1e-6.
    
    """
    rank = X.dim()
    unsqueezed = rank == 3
    if unsqueezed:
        X = X.unsqueeze(0)
    
    red = X[:, 1]
    nir = X[:, 0]
    
    ndvi = (nir - red) / (nir + red + 1e-6)
    
    if unsqueezed:
        return ndvi.squeeze(0)
    return ndvi



def calculate_gndvi(X: torch.Tensor, eps: float=1e-6):
    
    rank = X.dim()
    unsqueezed = rank == 3
    if unsqueezed:
        X = X.unsqueeze(0)
    
    green = X[:, 2]
    nir = X[:, 0]
    
    gndvi = (nir - green) / (nir + green + 1e-6)
    
    if unsqueezed:
        return gndvi.squeeze(0)
    return gndvi



def calculate_ndwi(X: torch.Tensor, eps: float=1e-6):
    
    rank = X.dim()
    unsqueezed = rank == 3
    if unsqueezed:
        X = X.unsqueeze(0)
    
    green = X[:, 2]
    nir = X[:, 0]
    
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    if unsqueezed:
        return ndwi.squeeze(0)
    return ndwi



def calculate_ngrdi(X: torch.Tensor, eps: float=1e-6):
    
    rank = X.dim()
    unsqueezed = rank == 3
    if unsqueezed:
        X = X.unsqueeze(0)
    
    green = X[:, 2]
    red = X[:, 1]
    
    ngrdi = (green - red) / (green + red + 1e-6)
    if unsqueezed:
        return ngrdi.squeeze(0)
    return ngrdi
import os
from typing import Tuple

def raise_if_not_exists(path: str) -> None:
    '''
    Raise a FileNotFoundError if the path does not exist.
    
    Parameters
    ----------
    path : str
        The path to check for existence.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path {path} does not exist.')

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    '''
    Convert a hex color to an RGB tuple.
    
    Parameters
    ----------
    hex_color : str
        The hex color to convert.
    
    Returns
    -------
    Tuple[int, int, int]
        The RGB tuple.
    '''
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
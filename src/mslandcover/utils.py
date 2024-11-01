import os

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

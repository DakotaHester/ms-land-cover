import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model',
        type=str,
        default='hrnet_w18',
        choices=['hrnet_w18', 'hrnet_w48'],
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='imagenet',
        choices=['imagenet', 'simclr', 'dae', 'hsv', 'simclr_hsv', 'simclr_dae', 'dae_hsv', 'simclr_dae_hsv'],
    )
    
    parser.add_argument(
        '--n-layers',
        type=int,
        default=1,
    )
    
    args = parser.parse_args()
    
    if args.n_layers < 1:
        parser.error('--n-layers must be greater than or equal to 1')
    
    return parser.parse_args()

def main() -> None:
    pass

if __name__ == '__main__':
    main()
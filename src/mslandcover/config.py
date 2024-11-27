from .utils import hex_to_rgb

# Mississippi Transverse Mercator projection parameters (https://epsg.io/3813)
MSTM_PROJ4 = '+proj=tmerc +lat_0=32.5 +lon_0=-89.75 +k=0.9998335 +x_0=500000 +y_0=1300000'
MSTM_WKT = 'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",32.5],PARAMETER["central_meridian",-89.75],PARAMETER["scale_factor",0.9998335],PARAMETER["false_easting",500000],PARAMETER["false_northing",1300000]'
MSTM_EPSG = '3813'

LEGEND_CLASSES = {
    0: 'Nodata',
    1: 'Open Water',
    2: 'Impervious (raised structures)',
    3: 'Impervious (paved surfaces)',
    4: 'Barren Land',
    5: 'Forest',
    6: 'Herbaceous',
    7: 'Cultivated Crops',
    8: 'Unclassified',
}

LEGEND_COLORS_HEX = {
    0: '#ffffff',
    1: '#516c97',
    2: '#989899',
    3: '#ffffff',
    4: '#805944',
    5: '#2a5439',
    6: '#88e5b8',
    7: '#f6c376',
    8: '#000000',
}

LEGEND_COLORS_RGB = {k: hex_to_rgb(v) for k, v in LEGEND_COLORS_HEX.items()}

# modified from HRNET w48 (https://github.com/HRNet/HRNet-Image-Classification/blob/master/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml)
HRNET_BASE_CONFIG = {
    'STAGE1': {
        'NUM_MODULES': 1,
        'NUM_RANCHES': 1,
        'BLOCK': 'BOTTLENECK',
        'NUM_BLOCKS': [4],
        'NUM_CHANNELS': [64],
        'FUSE_METHOD': 'SUM'
    },
    'STAGE2': {
        'NUM_MODULES': 1,
        'NUM_BRANCHES': 2,
        'BLOCK': 'BASIC',
        'NUM_BLOCKS': [4, 4],
        'NUM_CHANNELS': [48, 96],
        'FUSE_METHOD': 'SUM'
    },
    'STAGE3': {
        'NUM_MODULES': 4,
        'NUM_BRANCHES': 3,
        'BLOCK': 'BASIC',
        'NUM_BLOCKS': [4, 4, 4],
        'NUM_CHANNELS': [48, 96, 192],
        'FUSE_METHOD': 'SUM'
    },
    'STAGE4': {
        'NUM_MODULES': 3,
        'NUM_BRANCHES': 4,
        'BLOCK': 'BASIC',
        'NUM_BLOCKS': [4, 4, 4, 4],
        'NUM_CHANNELS': [48, 96, 192, 384],
        'FUSE_METHOD': 'SUM'
    },
    'IMAGE_DECODER': {
        'NUM_BLOCKS': 2, # number of blocks per decoder layer - 2 is the default for simplicity
    },
    'SIMCLR_PROJECTION_HEAD': {
        'NUM_HIDDENS': 1, # number of hidden layers to use in the projection head
        'EMBED_DIM': 128,
    }
}
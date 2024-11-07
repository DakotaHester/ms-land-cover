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
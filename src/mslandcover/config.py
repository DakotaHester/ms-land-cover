from .utils import hex_to_rgb

# Mississippi Transverse Mercator projection parameters (https://epsg.io/3813)
MSTM_PROJ4 = '+proj=tmerc +lat_0=32.5 +lon_0=-89.75 +k=0.9998335 +x_0=500000 +y_0=1300000'
MSTM_WKT = 'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",32.5],PARAMETER["central_meridian",-89.75],PARAMETER["scale_factor",0.9998335],PARAMETER["false_easting",500000],PARAMETER["false_northing",1300000]'
MSTM_EPSG = '3813'

NLCD_LEVEL_I_LEGEND_CLASSES = {
    0: 'Nodata',
    1: 'Water',
    2: 'Developed',
    3: 'Barren',
    4: 'Forest',
    5: 'Shrubland',
    6: 'Herbaceous',
    7: 'Cultivated',
    8: 'Wetlands',
}

NLCD_LEVEL_I_LEGEND_COLORS_HEX = {
    0: '#000000',
    1: '#4A6A9E',
    2: '#FF0000',
    3: '#CCC6B8',
    4: '#317645',
    5: '#A68B34',
    6: '#99C547',
    7: '#FBF452',
    8: '#C0E2F7',
}

NLCD_LEVEL_I_LEGEND_COLORS_RGB = {k: hex_to_rgb(v) for k, v in NLCD_LEVEL_I_LEGEND_COLORS_HEX.items()}

MS_LEGEND_CLASSES = {
    0: 'Nodata',
    1: 'Open Water',
    2: 'Impervious (raised structures)',
    3: 'Impervious (paved surfaces)',
    4: 'Barren Land',
    5: 'Forest',
    6: 'Herbaceous',
    7: 'Cultivated Crops',
    8: 'Wetlands',
    9: 'Unclassified',
}
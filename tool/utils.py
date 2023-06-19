import numpy as np
from astropy.io import fits


# used to read fits 

def read_fits(path, is_show_info=False):
    data = fits.open(path)
    if is_show_info:
        info = data.info()
    header = data[0].header
    data = data[0].data
    return data, header

# Standardized images

def max_min_norm(img):
    max_val = np.max(img)
    min_val = np.min(img)
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm

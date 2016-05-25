import numpy as np

def rescale_0_255(img):
    img = img.astype(np.float)
    img -= img.min()
    img /= img.max()
    img *= 255
    return img.astype(np.uint8)

def grayscale_to_rgb(img):
    return np.tile(img, (3,1,1)).transpose([1,2,0]).astype(np.uint8)

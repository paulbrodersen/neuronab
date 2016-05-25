import numpy as np

def rescale_0_255(img):
    img = img.astype(np.float)
    img -= img.min()
    img /= img.max()
    img *= 255
    return img.astype(np.uint8)

def grayscale_to_rgb(img):
    return np.tile(img, (3,1,1)).transpose([1,2,0]).astype(np.uint8)

def handle_image_input(path_or_array):
    assert type(path_or_array) in (str, np.ndarray), \
        "Image not a path (i.e. a string) or a numpy array!"
    if type(path_or_array) is str:
        img = plt.imread(path_or_array)
    else: # rename
        img = path_or_array
    return img

def handle_grayscale_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    assert np.ndim(img) == 2, \
        "Dimensionality of image is not 2! Is it really a grayscale image and not an RGB or RGBA image?"
    if img.dtype != np.uint8:
        raise Warning("Image not of type numpy.uint8. Casting to numpy.uint8 ...")
        img = img.astype(np.uint8)
    return img

def handle_rgb_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    assert np.ndim(img) in (3,4), \
        "Dimensionality of image is not 3! Is it really an RGB or RGBA image and not a grayscale image?"
    return img

def handle_binary_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    assert np.ndim(img) == 2, \
        "Dimensionality of image is not 2! Is it really a binary image and not an RGB or RGBA image?"
    if img.dtype != np.bool:
        raise Warning("Image not of type numpy.bool. Casting to numpy.bool ...")
        img = img.astype(np.bool)
    return img

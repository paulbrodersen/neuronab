import numpy as np
import matplotlib.pyplot as plt

def rescale_0_255(img):
    img = img.astype(np.float)
    img -= img.min()
    img /= img.max()
    img *= 255
    return img.astype(np.uint8)

def imcomplement(img):
    if img.dtype == np.bool:
        complement = np.logical_not(img)
    elif img.dtype == np.uint8:
        complement = 255 - img
    else:
        print "Warning: Image type not uint8 or bool. Assuming maximum pixel intensity of 255 when computing the image complement."
        complement = 255 - img
    return complement

def grayscale_to_rgb(img):
    return np.tile(img, (3,1,1)).transpose([1,2,0]).astype(np.uint8)

def handle_image_input(path_or_array):
    assert isinstance(path_or_array, (str, unicode, np.ndarray)), \
        "Image not a path (i.e. a string) or a numpy array! Currently, type(path_or_array) = {}.".format(type(path_or_array))
    if isinstance(path_or_array, (str, unicode)):
        img = plt.imread(path_or_array)
    else: # rename
        img = path_or_array
    return img

def handle_grayscale_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    assert np.ndim(img) == 2, \
        "Dimensionality of image is not 2! Is it really a grayscale image and not an RGB or RGBA image? Currently, ndim(img) = {}.".format(np.ndim(img))
    # if img.dtype != np.uint8:
    #     print Warning("Image not of type numpy.uint8. Casting to numpy.uint8 ...")
    #     img = img.astype(np.uint8)
    return img

def handle_rgb_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    assert np.ndim(img) in (3,4), \
        "Dimensionality of image is not 3! Is it really an RGB or RGBA image and not a grayscale image? Currently, ndim(img) = {}.".format(np.ndim(img))
    return img

def handle_binary_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    assert np.ndim(img) == 2, \
        "Dimensionality of image is not 2! Is it really a binary image and not an RGB or RGBA image? Currently, ndim(img) = {}.".format(np.ndim(img))
    if img.dtype != np.bool:
        print Warning("Image not of type numpy.bool. Casting to numpy.bool ...")
        img = img.astype(np.bool)
    return img

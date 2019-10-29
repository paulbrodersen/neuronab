#!/usr/bin/env python

import warnings
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
        warnings.warn("Image type not uint8 or bool. Assuming maximum pixel intensity of 255 when computing the image complement.")
        complement = 255 - img
    return complement

def grayscale_to_rgb(img):
    return np.tile(img, (3,1,1)).transpose([1,2,0]).astype(np.uint8)

def handle_image_input(path_or_array):
    assert isinstance(path_or_array, (str, np.ndarray)), \
        "Image not a path (i.e. a string) or a numpy array! Currently, type(path_or_array) = {}.".format(type(path_or_array))
    if isinstance(path_or_array, str):
        img = plt.imread(path_or_array)
    else: # rename
        img = path_or_array
    return img

def handle_grayscale_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    if img.ndim == 2:
        pass
    elif img.ndim == 3:
        warnings.warn("Grayscale image has 3 dimensions instead of the expected 2. Collapsing the last dimension by summing, ignoring the alpha channel.")
        img = np.sum(img[:,:,:3], axis=-1)
    else:
        raise ValueError("Shape of image is {}, which is not a valid image dimensionality.".format(img.shape))
    return img

def handle_rgb_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    assert np.ndim(img) in (3,4), \
        "Dimensionality of image is not 3! Is it really an RGB or RGBA image and not a grayscale image? Currently, ndim(img) = {}.".format(np.ndim(img))
    return img

def handle_binary_image_input(path_or_array):
    img = handle_image_input(path_or_array)
    if img.ndim == 2:
        pass
    elif img.ndim == 3:
        warnings.warn("Binary image has 3 dimensions instead of the expected 2. Collapsing the last dimension by summing, ignoring the alpha channel.")
        img = np.sum(img[:,:,:3], axis=-1)
    else:
        raise ValueError("Shape of image is {}, which is not a valid image dimensionality.".format(img.shape))

    if img.dtype != np.bool:
        warnings.warn("Image not of type numpy.bool. Casting to numpy.bool ...")
        img = img.astype(np.bool)
    return img

def plot_image_collection(images, titles, cmap='gray', *subplots_args, **subplots_kwargs):
    fig, axes = plt.subplots(sharex=True, sharey=True, *subplots_args, **subplots_kwargs)
    for img, title, ax in zip(images, titles, axes.ravel()):
        plot_image(img, title, ax, cmap)
    fig.tight_layout()
    return fig

def plot_image(img, title, ax, cmap='gray'):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize='small')
    ax.set_adjustable('box')
    ax.tick_params(
        axis        = 'both',
        which       = 'both',
        bottom      = False,
        top         = False,
        right       = False,
        left        = False,
        labelbottom = False,
        labelleft   = False,
    )

def count_objects(binary_mask):
    label_objects, nb_labels = scipy.ndimage.label(binary_mask)
    count = label_objects.max()
    return count

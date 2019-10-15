#!/usr/bin/env python

import numpy as np
import scipy.ndimage
import skimage.morphology as sm

from neuronab.utils import rescale_0_255, imcomplement

def remove_small_objects(binary_img, size_threshold=100):
    label_objects, nb_labels = scipy.ndimage.label(binary_img)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes >= size_threshold
    mask_sizes[0] = 0
    clean = mask_sizes[label_objects]
    return clean

def remove_large_objects(binary_img, size_threshold=100):
    label_objects, nb_labels = scipy.ndimage.label(binary_img)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes <= size_threshold
    mask_sizes[0] = 0
    clean = mask_sizes[label_objects]
    return clean

# def morphological_cleaning(img, disk):
def morphological_cleaning(img, selem):
    # disk = sm.disk(disk_size)
    eroded = sm.erosion(img, selem)
    reconstructed = sm.reconstruction(eroded, img)
    dilated = sm.dilation(reconstructed, selem)
    clean_complement = sm.reconstruction(imcomplement(dilated), imcomplement(reconstructed))
    clean = imcomplement(clean_complement)
    return rescale_0_255(clean)

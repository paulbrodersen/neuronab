import numpy as np
import scipy.ndimage
import skimage.morphology as sm

from utils import rescale_0_255

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

def imcomplement(img):
    complement = np.zeros_like(img)

    if img.dtype == np.bool:
        complement = np.logical_not(img)

    elif img.dtype == np.uint8:
        complement = 255 - img

    else:
        print "Warning: image type not uint8 or bool."
        print "Assuming maximum pixel intensity of 255..."
        complement = 255 - img

    return complement

def morphological_cleaning(img, disk_size=2):

    disk = sm.disk(disk_size)
    eroded = sm.erosion(img, disk)
    reconstructed = sm.reconstruction(eroded, img)
    dilated = sm.dilation(reconstructed, disk)

    cleaned = sm.reconstruction(imcomplement(dilated), imcomplement(reconstructed))

    return rescale_0_255(imcomplement(cleaned))

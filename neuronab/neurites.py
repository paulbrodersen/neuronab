#!/usr/bin/env python

import numpy as np

from skimage.morphology import disk, medial_axis
from skimage.filters import rank, meijering
from skimage.measure import label
from skimage.color import label2rgb

from neuronab import utils


def get_mask(neurite_marker,
             sigmas=range(1, 5),
             intensity_threshold=75.,
             size_threshold=1000,
             show=True):
    """Given a neurite stain image, returm a boolean mask that evaluates
    to true where there is a neurite. We carry out the following steps:

    1) Apply rank equalisation to obtain an evenly illuminated image.
    2) Apply the filter by Meijering et al. to isolate tubular structures.
    3) Obtain a mask by applying an intensity threshold.
    4) Remove small objects to clean up the mask.

    Arguments:
    ----------
    neurite_marker: string or numpy.uint8 array
        path to image of neurite marker OR
        corresponding grayscale image with intensities in the range (0-255)

    sigmas: iterable of floats (default range(1,5))
        Meijering filter scales.

    intensity_threshold: float (default 75.)
        Intensity threshold in percent.
        Applied after the Meijering filter to obtain a mask from the grayscale
        image.

    size_threshold: int (default 1000)
        Object size threshold in pixels.
        Smaller objects are removed from the mask.

    show: bool (default True)
        if True, plots intermediate steps of image analysis

    Returns:
    --------
    neurite_mask: ndarray
        Grayscale image indicating the presence of neurites.
        Pixel intensity scales approximately with the certainty that the pixel
        is part of a neurites.

    References:
    -----------
    Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,
    Unser, M. (2004). Design and validation of a tool for neurite
    tracing and analysis in fluorescence microscopy images. Cytometry
    Part A, 58(2), 167-176. DOI:10.1002/cyto.a.20022

    """
    # handle input
    raw = utils.handle_grayscale_image_input(neurite_marker)

    # equalise
    equalised = rank.equalize(raw, selem=disk(50))

    # apply meijering filter
    filtered = _apply_meijering_filter(equalised, sigmas)

    # threshold
    thresholded = filtered > np.percentile(filtered, intensity_threshold)

    # remove small objects in the image
    clean = _remove_small_objects(thresholded, size_threshold)

    if show:
        combined = raw.astype(np.float)
        combined[clean] *= 1.15

        titles = [
            'Neurite marker'               ,
            'Local histogram equalisation' ,
            'Meijering ridge filter'       ,
            'Thresholded'                  ,
            'Removed small objects'        ,
            'Combined'                     ,
        ]

        images = [raw         ,
                  equalised   ,
                  filtered    ,
                  thresholded ,
                  clean       ,
                  combined    ,
        ]

        fig1 = utils.plot_image_collection(images, titles, nrows=2, ncols=3)
        fig2 = utils.plot_image_mask_comparison(raw, clean)

    return clean


def _apply_meijering_filter(image, sigmas):
    smoothed = rank.mean_percentile(iamage, disk(5), p0=0.25, p1=0.75)
    filtered = meijering(smoothed, sigmas=sigmas, black_ridges=False)

    # Meijering filter always evaluates to high values at the image frame;
    # we hence set the filtered image to zero at those locations
    frame = np.ones_like(filtered, dtype=np.bool)
    d = 2 * np.max(sigmas) + 1
    frame[d:-d, d:-d] = False
    filtered[frame] = np.min(filtered)

    return filtered


def _remove_small_objects(binary_mask, size_threshold):
    label_image = label(binary_mask)
    object_sizes = np.bincount(label_image.ravel())
    labels2keep, = np.where(object_sizes > size_threshold)
    labels2keep = labels2keep[1:] # remove the first label, which corresponds to the background
    clean = np.in1d(label_image.ravel(), labels2keep).reshape(label_image.shape)
    return clean


def get_length(neurite_mask, show=False):
    """
    Arguments:
    ----------
        neurite_mask: string or numpy.bool array
            path to binary image indicating the presence of neurites, OR
            corresponding boolean numpy.ndarray

        show: bool (default True)
            if True, plots intermediate steps of image analysis

    Returns:
    --------
        neurite_length: int
            total neurite length in pixels
    """

    neurite_mask = utils.handle_binary_image_input(neurite_mask)
    neurite_skeleton = medial_axis(neurite_mask)
    neurite_length = neurite_skeleton.sum()

    if show:
        images = [neurite_mask, neurite_skeleton]
        titles = ['Neurite mask', 'Medial axis']
        fig = utils.plot_image_collection(images, titles, nrows=1, ncols=2)
        fig.suptitle('Neurite length', fontsize='large')

    return neurite_length

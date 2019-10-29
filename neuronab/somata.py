#!/usr/bin/env python
"""
Isolate and count neuronal somata.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import peak_widths
from scipy.stats import gaussian_kde

from skimage.morphology import (
    disk,
    opening,
    closing,
    binary_closing,
    binary_opening,
    watershed,
)
from skimage.feature import peak_local_max, blob_log
from skimage.measure import label
from skimage.color import label2rgb
from skimage.filters import rank

from phasepack import phasesym
from neuronab import (
    cleaning,
    utils
)


def get_mask(soma_marker, intensity_threshold=50., size_threshold=50, show=True):
    """Get a binary mask indicating the presence or absence of soma in a
    (immuno-)fluorescence image of a somatic marker.

    Arguments:
    ----------
    soma_marker: string or numpy.uint8 array
        Path to image of soma marker OR
        corresponding grayscale image with intensities in the range (0-255).

    intensity_threshold: float (default 50.)
        Minimum pixel intensity in percent which are considered for the soma mask.

    size_threshold : int (default 50)
        Minimum object size in pixels above which objects are included in the soma mask.

    show: bool (default True)
        If True, plots intermediate steps of image analysis.

    Returns:
    --------
    soma_mask: numpy.bool array
        Binary image indicating the presence of somata.

    """
    # handle input
    raw = utils.handle_grayscale_image_input(soma_marker)
    equalised = rank.equalize(raw, selem=disk(50))
    intensity_threshold = np.percentile(raw, intensity_threshold)
    equalised[raw < intensity_threshold] = intensity_threshold

    # determine local phase-symmetry -> maxima correspond to soma
    phase, _, _, _ = phasesym(equalised,
                              nscale        = 10,
                              norient       = 2,
                              minWaveLength = 0.5,
                              mult          = 3.,
                              sigmaOnf      = 0.55,
                              k             = 2.,
                              polarity      = 1,
                              noiseMethod   = -1)
    phase = utils.rescale_0_255(phase)

    # morphological cleaning
    clean = cleaning.morphological_cleaning(phase, disk(3))

    # remove thin and small objects
    mask = binary_opening(clean > 0, disk(2))
    mask = cleaning.remove_small_objects(mask, size_threshold)
    clean = mask * clean

    if show:
        images = [raw, equalised, phase, clean]
        titles = ['Soma marker', 'Local histogram equalisation', 'Phase symmetry', 'Morphological cleaning']
        utils.plot_image_collection(images, titles, nrows=2, ncols=2)

    return clean


def get_count(soma_marker, soma_mask=None, threshold=0.1, show=True):
    # estimate bounds on soma area in pixels
    if soma_mask is None:
        soma_mask = get_mask(soma_marker, show=show) > 0
    mode, (lower, upper) = _get_soma_size(soma_mask, show=show)

    # fit blobs; constrain fit using bounds on soma size
    blobs = _detect_blobs(soma_marker, lower, upper, threshold, show=show)
    return len(blobs), blobs


def _get_soma_size(soma_mask, show=True):
    object_sizes = _get_object_sizes(soma_mask)
    mode, (lower, upper) = _get_mode(object_sizes,
                                     return_bounds=True,
                                     show=show)
    return mode, (lower, upper)


def _get_object_sizes(binary_mask):
    labelled = label(binary_mask)
    object_sizes = np.bincount(labelled.ravel())
    # first object corresponds to the background
    object_sizes = object_sizes[1:]
    return object_sizes


def _get_mode(values, return_bounds=True, show=True):
    """Estimate mode of a distribution using Gaussian kernel density estimation.

    Arguments:
    ----------
    values : iterable of floats
        Samples from the empirical distribution.
    return_bounds : bool (default True)
        If true, also return the values corresponding to the width-at-half-height of the mode peak.
    show : bool (default True)
        If True, plot a histogram with the density estimate, the mode, and bounds.

    Returns:
    --------
    mode : float
    bounds : (float, float)
    """

    # sanitise input
    invalid = np.logical_or(np.isnan(values), np.isinf(values))
    if np.any(invalid):
        import warnings
        warnings.warn("Some values are either NaN or inf. These will be ignored in the computation.")
        values = values[~invalid]

    # fit distribution
    kde = gaussian_kde(values)
    min_value = np.min(values)
    max_value = np.max(values)
    resolution = 1000
    x = np.linspace(min_value, max_value, resolution)
    density = kde.evaluate(x)

    # determine its mode
    mode_idx = np.argmax(density)

    # determine lower and upper bound
    width, width_height, left_ips, right_ips = peak_widths(density, [mode_idx])
    left_idx, = left_ips
    right_idx, = right_ips

    # convert from index space to value space
    def idx2val(idx):
        return min_value + (max_value - min_value) / resolution * idx

    mode  = idx2val(mode_idx)
    left  = idx2val(left_idx)
    right = idx2val(right_idx)

    if show:
        fig, ax = plt.subplots(1, 1)
        ax.hist(values, bins=50, density=True)
        ax.plot(x, density)
        for value in [mode, left, right]:
            ax.axvline(value,  color='gray', linestyle='--')
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')

    if return_bounds:
        return mode, (left, right)
    else:
        return mode


def _detect_blobs(image, min_size=50, max_size=200, threshold=0.2, show=True):
    """
    Thin wrapper around skimage.morphology.blob_log.

    Finds blobs in the given grayscale image.
    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : path or 2D ndarray
        Path to grayscale image or the image itself.
        Blobs are assumed to be light on dark background (white on black).
    min_size : float
        Minimum blob area in pixels.
    max_size : float
        Maximum blob area in pixels.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    show: bool

    Returns
    -------
    blobs : iterable of (x, y, radius) tuples
        The (x, y) positions and radii of the detected blobs.
    """

    image = utils.handle_grayscale_image_input(image)

    # convert area to an estimate of the STD of a corresponding Gaussian kernel
    # \sigma = radius / \sqrt(2)
    min_sigma = np.sqrt(min_size / np.pi) / np.sqrt(2)
    max_sigma = np.sqrt(max_size / np.pi) / np.sqrt(2)

    # detect blobs
    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)

    # convert sigma estimates back to radii
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    if show:
        fig, ax = plt.subplots(1,1)
        ax.imshow(image, cmap='gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='yellow', linewidth=1., fill=False)
            ax.add_patch(c)

    return blobs

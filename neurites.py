#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import skimage.morphology

from skimage import draw
from skimage.filters import rank
from skimage.transform import probabilistic_hough_line

import phasepack.phasepack as pp
import cleaning
import utils

def get_mask(neurite_marker, show=True, save=None):
    """
    Arguments:
    ----------
        neurite_marker: string or numpy.uint8 array
            path to image of neurite marker OR
            corresponding grayscale image with intensities in the range (0-255)

        show: bool (default True)
            if True, plots intermediate steps of image analysis

        save: str (default None)
    `      if not None (and show is True), figures will be saved under save+<1>.pdf

    Returns:
    --------
        neurite_mask: numpy.bool array
            binary image indicating the presence of neurites

    """
    # handle input
    raw = utils.handle_grayscale_image_input(neurite_marker)

    selem = skimage.morphology.disk(50)
    equalised = rank.equalize(raw, selem=selem)

    # determine local phase-symmetry -> maxima correspond to neurite
    phase = pp.phasesym(equalised,
                        nscale        = 5,
                        norient       = 3,
                        minWaveLength = 1.,
                        mult          = 3.,
                        sigmaOnf      = 0.55,
                        k             = 1.,
                        polarity      = 1,
                        noiseMethod   = -1)[0]
    phase = utils.rescale_0_255(phase)

    # Morphological cleaning using 1-connectivity:
    # clean with a vertical/horizontal and diagonal cross individually,
    # and then combine the results.
    # This leverages the elongated nature of the neurites.
    # Use the combination of both crosses (i.e. a square) yields inferior results.
    selem = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    clean_1 = cleaning.morphological_cleaning(phase, selem)
    selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    clean_2 = cleaning.morphological_cleaning(phase, selem)
    clean = clean_1 + clean_2

    # get neurite skeleton
    skeleton = _skeletonize(clean)

    # connect isolated pieces by finding straight lines through them
    connected = _connect_broken_lines(skeleton, threshold=1, line_length=50, line_gap=25)

    # using skeleton as a seed, reconstruct the cleaned phase image
    reconstructed = _reconstruct(clean, connected)

    # threshold to binary mask
    binary = reconstructed > 0

    # close elements
    disk = skimage.morphology.disk(3)
    closed = skimage.morphology.closing(binary, disk)

    # remove isolated blobs
    neurite_mask = cleaning.remove_small_objects(closed, size_threshold=100)

    if show:
        titles = [
            'Neurite marker'                 ,
            'Local histogram equalisation'   ,
            'Phase symmetry'                 ,
            'Morphological cleaning'         ,
            'Hough line transform'           ,
            'Reconstruction and thresholded' ,
        ]

        images = [raw,
                  equalised,
                  phase,
                  clean,
                  connected,
                  neurite_mask,
        ]

        fig = utils.plot_image_collection(images, titles, nrows=2, ncols=3)

        if save != None:
            fig.savefig(save + '{}.pdf'.format(1), dpi=300)

    return neurite_mask


def _skeletonize(binary_image):
    return skimage.morphology.medial_axis(binary_image)


def _connect_broken_lines(broken, threshold=1, line_length=30, line_gap=20):
    lines = probabilistic_hough_line(broken,
                                     threshold,
                                     line_length,
                                     line_gap)
    connected = np.zeros_like(broken, dtype=np.uint8)
    for line in lines:
        p0, p1 = line
        rr, cc = draw.line(p0[1], p0[0], p1[1], p1[0])
        connected[rr, cc] += 1

    return connected


def _reconstruct(neurites, skeleton, show=False):

    # reconstruct neurites from the skeleton outwards;
    # seed value for reconstruct cannot exceed the pixel value in the neurite image
    seed = np.zeros_like(neurites)
    seed[skeleton > 0] = neurites[skeleton > 0]
    reconstructed = skimage.morphology.reconstruction(seed, neurites)

    # use skeleton value in regions where the pixel value in the neurite image
    # is smaller than in the skeleton image;
    holes = np.zeros_like(neurites)
    fill_value = np.median(neurites[neurites > 0])
    holes[skeleton > neurites] = fill_value
    combined = reconstructed + holes

    if show:
        images = [neurites, skeleton, reconstructed, combined]
        titles = ['Neurite mask', 'Medial axis', 'Reconstructed', 'Combined']
        fig = utils.plot_image_collection(images, titles, nrows=2, ncols=2)

    return utils.rescale_0_255(combined)


def get_length(neurite_mask, show=False, save=None):
    """
    Arguments:
    ----------
        neurite_mask: string or numpy.bool array
            path to binary image indicating the presence of neurites, OR
            corresponding boolean numpy.ndarray

        show: bool (default True)
            if True, plots intermediate steps of image analysis

        save: str (default None)
    `      if not None (and show is True), figures will be saved under save+<2>.pdf

    Returns:
    --------
        neurite_length: int
            total neurite length in pixels
    """

    neurite_mask = utils.handle_binary_image_input(neurite_mask)
    neurite_skeleton = _skeletonize(neurite_mask)
    neurite_length = neurite_skeleton.sum()

    if show:
        images = [neurite_mask, neurite_skeleton]
        titles = ['Neurite mask', 'Medial axis']
        fig = utils.plot_image_collection(images, titles, nrows=1, ncols=2)
        fig.suptitle('Neurite length', fontsize='large')
        if save != None:
            fig.savefig(save + '{}.pdf'.format(2), dpi=300)

    return neurite_length

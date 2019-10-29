#!/usr/bin/env python

import numpy as np

from skimage.morphology import (
    binary_dilation,
    disk,
)

from neuronab import (
    cleaning,
    utils,
    neurites,
)


def get_mask(synaptic_marker, neurite_mask,
             min_synapse_size=25,
             max_synapse_size=400,
             min_synapse_brightness=97.5,
             show=False):

    """Given a grayscale image of a synaptic stain, and a boolean mask for
    neurites, determine the pixels corresponding to synapses in the following steps:

    1) Only bright pixels can contain a synapse. Hence threshold the
    synaptic stain based on brightness.

    2) Synapses have a certain size. Hence remove objects that are
    either too small or too large.

    3) Synapses are on neurites. Hence remove all synapse candidates
    that are not on -- or the immediate vicinity of -- a neurite.

    Arguments:
    ----------
    synaptic_marker: string or numpy.uint8 array
        path to grayscale image of synaptic marker, OR
        corresponding numpy array with values in the range (0-255)

    neurite_mask: string or numpy.bool array
        path to binary image indicating the presence of neurites, OR
        corresponding boolean numpy.ndarray

    min_synapse_size: int (default 25)
        minimum acceptable synapse sizes in pixels

    max_synapse_size: int (default 400)
        maximum acceptable synapse sizes in pixels

    min_synapse_brightness: float in the range 0.-100., (default 95.)
        image intensity threshold in percent above which objects
        in synaptic marker images are labelled as putative synapses

    show: bool (default True)
        if True, plots intermediate steps of image analysis

    Returns:
    --------
    synapse_mask: numpy.bool array
        binary image indicating the presence of a synapse

    """

    # handle input
    synapses_raw = utils.handle_grayscale_image_input(synaptic_marker)
    neurite_mask = utils.handle_binary_image_input(neurite_mask)

    # threshold
    thresholded = synapses_raw > np.percentile(synapses_raw, min_synapse_brightness)

    # remove too large objects, remove too small objects
    cleaned = cleaning.remove_large_objects(thresholded, max_synapse_size+1)
    cleaned = cleaning.remove_small_objects(cleaned,     min_synapse_size-1)

    # restrict synapse candidates to puncta within or juxtaposed to the neurite mask;
    # dilate mask to catch synapses that are next to the neurite but not directly on it
    neurite_mask = binary_dilation(neurite_mask, disk(2))
    synapse_mask = np.logical_and(cleaned, neurite_mask)

    if show:
        images = [synapses_raw, thresholded, cleaned+0.3*thresholded, synapse_mask+0.3*neurite_mask]
        titles = ['Synaptic marker', 'Thresholded', 'Within size range', 'Within neurite mask']
        fig = utils.plot_image_collection(images, titles, nrows=2, ncols=2)

    return synapse_mask


def get_count(synapse_mask):
    """Count the number of putative synapses in the synapse mask.

    Arguments:
    ----------
    synapse_mask: numpy.bool array
        binary image indicating the presence of a synapse

    Returns:
    --------
    synapse_count : int
        number synapses
    """
    return utils.count_objects(synapse_mask)

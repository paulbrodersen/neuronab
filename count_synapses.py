import numpy as np
import matplotlib.pyplot as plt; plt.ion()

import scipy.ndimage
import skimage.morphology
import skimage.transform
import skimage.draw

import phasepack; reload(phasepack)
import cleaning; reload(cleaning)
import utils; reload(utils)

def run(neurite_marker,
        primary_synaptic_marker,
        secondary_synaptic_marker=None,
        range_synapse_size=(9, 100),
        minimum_synapse_brightness=95.,
        show=True):
    """
    Arguments:
    ----------
        neurite_marker: string or numpy.uint8 array
            path to image of neurite marker OR
            corresponding grayscale image with intensities in the range (0-255)

        primary_synaptic_marker: string or numpy.uint8 array
            path to image of synaptic marker OR
            corresponding grayscale image with intensities in the range (0-255)

        secondary_synaptic_marker: string or numpy.uint8 array (optional, default None)
            path to image of a secondary synaptic marker OR
            corresponding grayscale image with intensities in the range (0-255)

        range_synapse_size: 2-tuple of integers, default (1, 70)
            range of acceptable synapse sizes

        minimum_synapse_brightness: float in the range 0.-100., (default 95.)
            image intensity threshold in percent above which objects
            in synaptic marker images are labelled as putative synapses

        show: bool (default True)
            if True, plots intermediate steps of image analysis


    Returns:
    --------
        neurite_length: int
            length of medial axis of detected neurites in pixels

        primary_synaptic_marker_count: int
            number of synapses detected in image of primary synaptic marker

        secondary_synaptic_marker_count: int
            number of synapses detected in image of secondary synaptic marker;
            only returned if secondary_synaptic_marker is not None;

        dual_labelled: int
            number of synapses detected in images of both synaptic markers;
            only returned if secondary_synaptic_marker is not None;
    """

    # --------------------------------------------------------------------------------
    # manage input

    # check paths/images
    assert type(neurite_marker) in (str, np.ndarray), \
        "Argument neurite_marker not a path (i.e. a string) or an image (i.e. a numpy array)!"
    assert type(primary_synaptic_marker) in (str, np.ndarray), \
        "Argument primary_synaptic_marker not a path (i.e. a string) or an image (i.e. a numpy array)!"
    assert type(secondary_synaptic_marker) in (str, np.ndarray, type(None)), \
        "Argument secondary_synaptic_marker not a path (i.e. a string) or an image (i.e. a numpy array) or None!"

    # load data where applicable
    if type(neurite_marker) is str:
        neurites_raw = plt.imread(neurite_marker)
    else: # rename
        neurites_raw = neurite_marker

    if type(primary_synaptic_marker) is str:
        primary_raw = plt.imread(primary_synaptic_marker)
    else:
        primary_raw = primary_synaptic_marker

    if type(secondary_synaptic_marker) is str:
        secondary_raw = plt.imread(secondary_synaptic_marker)
    else: # rename
        secondary_raw = secondary_synaptic_marker

    # check that images are grayscale and not RGB or RGBA
    assert np.ndim(neurites_raw) == 2, \
        "Dimensionality of neurite marker image exceeds 2! Is it a grayscale image and not and RGB or RGBA image?"
    assert np.ndim(primary_raw) == 2, \
        "Dimensionality of primary synaptic marker image exceeds 2! Is it a grayscale image and not and RGB or RGBA image?"
    if secondary_raw != None:
        assert np.ndim(secondary_raw) == 2, \
            "Dimensionality of secondary synaptic marker image exceeds 2! Is it a grayscale image and not and RGB or RGBA image?"

    # --------------------------------------------------------------------------------
    # isolate neurites and determine neurite length

    neurites = _isolate_neurites(neurites_raw, show)
    neurite_length = _get_neurite_length(neurites, show)

    # --------------------------------------------------------------------------------
    # find synapse candidates and count

    primary = _isolate_synapses(primary_raw, neurites, range_synapse_size, minimum_synapse_brightness, show)
    primary_count = _count_synapses(primary)

    if secondary_raw == None:

        if show == True:
            combined = (neurites_raw).astype(np.float) # + neurites_raw
            combined -= combined.min()
            combined /= combined.max()
            combined *= 255
            combined = utils.grayscale_to_rgb(combined)
            combined[np.where(primary)] = np.array([255, 0, 0])

            fig = plt.figure()
            ax1 = fig.add_subplot(2,4,1)
            ax1.imshow(neurites_raw, cmap='gray')
            ax2 = fig.add_subplot(2,4,2)
            ax2.imshow(neurites, cmap='gray')
            ax3 = fig.add_subplot(2,4,5)
            ax3.imshow(primary_raw, cmap='gray')
            ax4 = fig.add_subplot(2,4,6)
            ax4.imshow(primary, cmap='gray')
            ax5 = fig.add_subplot(1,2,2)
            ax5.imshow(combined, cmap='gray')

            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.tight_layout()

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(primary_raw, cmap='gray')
            ax2.imshow(combined)
            for ax in [ax1, ax2]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.tight_layout()

        return neurite_length, primary_count

    else:
        secondary = _isolate_synapses(secondary_raw, neurites, range_synapse_size, minimum_synapse_brightness, show)
        secondary_count = _count_synapses(secondary)

        dual_labelled = _is_dual_labelled(primary, secondary, show)
        dual_labelled_count = _count_synapses(dual_labelled)

        if show == True:
            # combined = (primary_raw + secondary_raw + 2*neurites_raw)/20. \
            #            + primary.astype(np.float) * 255 + secondary.astype(np.float) * 255

            combined = (neurites_raw).astype(np.float)
            combined -= combined.min()
            combined /= combined.max()
            combined *= 255
            combined = utils.grayscale_to_rgb(combined)
            combined[np.where(primary)] = np.array([255, 0, 0])
            combined[np.where(secondary)] = np.array([0, 255, 0])

            fig = plt.figure()
            ax1 = fig.add_subplot(3,4,1)
            ax1.imshow(neurites_raw, cmap='gray')
            ax2 = fig.add_subplot(3,4,2)
            ax2.imshow(neurites, cmap='gray')

            ax3 = fig.add_subplot(3,4,5)
            ax3.imshow(primary_raw, cmap='gray')
            ax4 = fig.add_subplot(3,4,6)
            ax4.imshow(primary, cmap='gray')

            ax5 = fig.add_subplot(3,4,9)
            ax5.imshow(secondary_raw, cmap='gray')
            ax6 = fig.add_subplot(3,4,10)
            ax6.imshow(secondary, cmap='gray')

            ax7 = fig.add_subplot(1,2,2)
            ax7.imshow(combined)

            for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.tight_layout()

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(primary_raw + secondary_raw, cmap='gray')
            ax2.imshow(combined)
            for ax in [ax1, ax2]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.tight_layout()

        return neurite_length, primary_count, secondary_count, dual_labelled_count

def _isolate_neurites(img, show):
    """
    Arguments:
    ----------
        img: np.uint array
            gray-scale image of neurites

    Returns:
    --------
        neurites: np.bool array
            boolean mask of neurite pixels
    """

    # 1) determine local phase-symmetry -> maxima correspond to neurite
    phase = phasepack.phasesym(img,
                               nscale=5,
                               norient=3,
                               minWaveLength=1,
                               mult=2.1,
                               sigmaOnf=0.55,
                               k=1., # 5.,
                               polarity=1,
                               noiseMethod=-1)[0]
    phase = utils.rescale_0_255(phase)

    # 2) (morphological) cleaning
    clean = cleaning.morphological_cleaning(phase, disk_size=1)

    # 3) threshold to binary mask
    binary = clean > 1

    # 4) close image to connect isolated pieces of neurite
    disk = skimage.morphology.disk(6)
    closed = skimage.morphology.closing(binary, disk)

    # 5) dilate mask to catch synapses that are next to the neurite but not directly on it
    disk = skimage.morphology.disk(2)
    dilated = skimage.morphology.binary_dilation(closed, disk)

    # remove isolated pixels
    neurites = cleaning.remove_small_objects(dilated, size_threshold=64)

    if show == True:
        fig = plt.figure()
        ax1 = fig.add_subplot(2,4,1)
        ax1.imshow(img, cmap='gray')
        ax2 = fig.add_subplot(2,4,2)
        ax2.imshow(phase, cmap='gray')
        ax3 = fig.add_subplot(2,4,5)
        ax3.imshow(closed, cmap='gray')
        ax4 = fig.add_subplot(2,4,6)
        ax4.imshow(dilated, cmap='gray')
        ax5 = fig.add_subplot(1,2,2)
        ax5.imshow(neurites, cmap='gray')

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return neurites

def _get_neurite_length(neurites, show):
    neurite_skeleton = skeletonize(neurites)
    length = neurite_skeleton.sum()

    if show == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(neurites, cmap='gray')
        ax2.imshow(neurite_skeleton, cmap='gray')
        for ax in [ax1, ax2]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return length

def _isolate_synapses(synapses_raw,
                      neurites,
                      range_synapse_size,
                      minimum_synapse_brightness,
                      show):

    # 1) threshold
    thresholded = synapses_raw > np.percentile(synapses_raw, minimum_synapse_brightness)

    # 2) remove too large objects, remove too small objects
    cleaned = cleaning.remove_large_objects(thresholded, range_synapse_size[1]+1)
    cleaned = cleaning.remove_small_objects(cleaned,     range_synapse_size[0]-1)

    # 3) restrict synapse candidates to puncta within the neurite mask
    synapses = np.logical_and(cleaned, neurites)

    if show == True:
        cleaned = utils.rescale_0_255(cleaned) + 50 * neurites

        images = [synapses_raw, thresholded, cleaned, synapses]
        fig, axes = plt.subplots(2,2)
        for img, ax in zip(images, axes.ravel()):
            # ax.imshow(img, cmap='gray')
            ax.imshow(img, cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return synapses

def _count_synapses(synapses):
    label_objects, nb_labels = scipy.ndimage.label(synapses)
    count = label_objects.max()
    return count

def _is_dual_labelled(primary, secondary, show):
    dual_labelled = np.logical_and(primary, secondary)

    if show == True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(primary, cmap='gray')
        ax2.imshow(secondary, cmap='gray')
        ax3.imshow(dual_labelled, cmap='gray')
        for ax in [ax1, ax2, ax3]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return dual_labelled

def skeletonize(binary_image):
    return skimage.morphology.medial_axis(binary_image)

def _connect_broken_lines(broken, show):

    # approach 1: skeletonize, find endpoints, join fragments
    # skel = skeleton.skeletonize(binary_image)
    # endpts = skeleton.find_end_points(skel)
    # TODO: connect endpoints in a smart way

    # approach 2: striaght line Hough transform
    lines = skimage.transform.probabilistic_hough_line(broken,
                                                       threshold=1,
                                                       line_length=30,
                                                       line_gap=20)
    connected = np.zeros_like(broken, dtype=np.uint8)
    for line in lines:
        p0, p1 = line
        rr, cc = skimage.draw.line(p0[1], p0[0], p1[1], p1[0])
        connected[rr, cc] += 1

    return connected

def test_isolate_neurites(paths):
    for path in paths:
        img = plt.imread(path)
        _isolate_neurites(img, show=True)
    return

def test_connect_broken_lines(paths):
    for path in paths:
        neurites_raw = plt.imread(path)
        neurites = _isolate_neurites(neurites_raw, show=False)
        neurite_skeleton = skeleton.skeletonize(neurites)
        connected = _connect_broken_lines(neurite_skeleton, show=False)

        # 4) close image to connect isolated pieces of neurite
        disk = skimage.morphology.disk(2)
        closed = skimage.morphology.closing(connected > 0, disk)

        fig, axes = plt.subplots(2,2)
        axes = axes.ravel()
        axes[0].imshow(neurites_raw, cmap='gray')
        axes[1].imshow(neurites, cmap='gray')
        axes[2].imshow(connected, cmap='gray')
        axes[3].imshow(closed, cmap='gray')

        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return

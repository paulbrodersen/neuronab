import numpy as np
import matplotlib.pyplot as plt; plt.ion(); plt.close('all')

import scipy.ndimage
import skimage.morphology
import skimage.transform
import skimage.draw

import phasepack.phasepack as pp; reload(pp)
import cleaning; reload(cleaning)
import utils; reload(utils)

global TITLE_FONT_SIZE
TITLE_FONT_SIZE = 'large'

def isolate(neurite_marker, show):
    """
    Arguments:
    ----------
        neurite_marker: string or numpy.uint8 array
            path to image of neurite marker OR
            corresponding grayscale image with intensities in the range (0-255)

        show: bool (default True)
            if True, plots intermediate steps of image analysis

    Returns:
    --------
        neurite_mask: numpy.bool array
            binary image indicating the presence of neurites

    """
    # handle input
    raw = utils.handle_grayscale_image_input(neurite_marker)

    # determine local phase-symmetry -> maxima correspond to neurite
    phase = pp.phasesym(raw,
                        nscale=5,
                        norient=3,
                        minWaveLength=1,
                        mult=2.1,
                        sigmaOnf=0.55,
                        k=1., # 5.,
                        polarity=1,
                        noiseMethod=-1)[0]
    phase = utils.rescale_0_255(phase)

    # morphological cleaning
    clean = cleaning.morphological_cleaning(phase, disk_size=1)

    # threshold to binary mask
    binary = clean > 0

    # close image to connect isolated pieces of neurite
    disk = skimage.morphology.disk(10)
    closed = skimage.morphology.closing(binary, disk)

    # remove isolated blobs
    neurite_mask = cleaning.remove_small_objects(closed, size_threshold=64)

    if show == True:
        fig, axes = plt.subplots(2,2)
        ax1, ax2, ax3, ax4 = axes.ravel()

        fig.suptitle('Neurite isolation', fontsize=TITLE_FONT_SIZE)

        ax1.imshow(raw, cmap='gray')
        ax1.set_title('input image')

        ax2.imshow(phase, cmap='gray')
        ax2.set_title('phase symmetry')

        ax3.imshow(clean, cmap='gray')
        ax3.set_title('morphological cleaning')

        ax4.imshow(neurite_mask, cmap='gray')
        ax4.set_title('threshold & morphological closing')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        fig.tight_layout()

    return neurite_mask

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
    neurite_skeleton = _skeletonize(neurite_mask)
    neurite_length = neurite_skeleton.sum()

    if show == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Neurite length', fontsize=TITLE_FONT_SIZE)

        ax1.imshow(neurite_mask, cmap='gray')
        ax1.set_title('binary mask')

        ax2.imshow(neurite_skeleton, cmap='gray')
        ax2.set_title('medial axis')

        for ax in [ax1, ax2]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return neurite_length

def _skeletonize(binary_image):
    return skimage.morphology.medial_axis(binary_image)

def _connect_broken_lines(broken, show):
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

# def test_isolate_neurites(paths):
#     for path in paths:
#         img = plt.imread(path)
#         _isolate_neurites(img, show=False)
#     return

# def test_connect_broken_lines(paths):
#     for path in paths:
#         neurites_raw = plt.imread(path)
#         neurites = _isolate_neurites(neurites_raw, show=False)
#         neurite_skeleton = skeleton.skeletonize(neurites)
#         connected = _connect_broken_lines(neurite_skeleton, show=False)

#         # 4) close image to connect isolated pieces of neurite
#         disk = skimage.morphology.disk(2)
#         closed = skimage.morphology.closing(connected > 0, disk)

#         fig, axes = plt.subplots(2,2)
#         axes = axes.ravel()
#         axes[0].imshow(neurites_raw, cmap='gray')
#         axes[1].imshow(neurites, cmap='gray')
#         axes[2].imshow(connected, cmap='gray')
#         axes[3].imshow(closed, cmap='gray')

#         for ax in axes:
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#         fig.tight_layout()

#     return

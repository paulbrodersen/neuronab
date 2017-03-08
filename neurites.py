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

def isolate(neurite_marker, show=True, save=None):
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
    # selem = skimage.morphology.square(3)
    # selem = skimage.morphology.disk(1)
    selem = np.array([[1,0,1],[0,1,0],[1,0,1]])
    clean_1 = cleaning.morphological_cleaning(phase, selem)

    selem = np.array([[0,1,0],[1,1,1],[0,1,0]])
    clean_2 = cleaning.morphological_cleaning(phase, selem)
    clean = clean_1 + clean_2

    # get neurite skeleton
    neurite_skeleton = _skeletonize(clean)

    # connect isolated pieces by finding straight lines through them
    connected = _connect_broken_lines(neurite_skeleton)

    # using skeleton as a seed, reconstruct the cleaned phase image
    reconstructed = _reconstruct(clean, connected)

    # threshold to binary mask
    binary = reconstructed > 0

    # # thicken neurites
    # disk = skimage.morphology.disk(10)
    # closed = skimage.morphology.closing(binary, disk)

    # # remove isolated blobs
    # neurite_mask = cleaning.remove_small_objects(closed, size_threshold=64)

    neurite_mask = binary

    if show == True:
        fig, axes = plt.subplots(2,3)
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

        # fig.suptitle('Neurite isolation', fontsize=TITLE_FONT_SIZE)

        ax1.imshow(raw, cmap='gray')
        ax1.set_title('input image')

        ax2.imshow(phase, cmap='gray')
        ax2.set_title('phase symmetry')

        ax3.imshow(clean, cmap='gray')
        ax3.set_title('morphological cleaning')

        ax4.imshow(connected, cmap='gray')
        ax4.set_title('neurite skeleton')

        ax5.imshow(reconstructed, cmap='gray')
        ax5.set_title('reconstruction')

        ax6.imshow(neurite_mask, cmap='gray')
        ax6.set_title('thresholded mask')

        for ax in axes.ravel():
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        fig.tight_layout()

        if save != None:
            fig.savefig(save + '{}.pdf'.format(1), dpi=300)

    return neurite_mask

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

        if save != None:
            fig.savefig(save + '{}.pdf'.format(2), dpi=300)

    return neurite_length

def _skeletonize(binary_image):
    return skimage.morphology.medial_axis(binary_image)

def _connect_broken_lines(broken):
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
        fig, axes = plt.subplots(2,2)
        axes = axes.ravel()
        axes[0].imshow(neurites, cmap='gray')
        axes[1].imshow(skeleton, cmap='gray')
        axes[2].imshow(reconstructed, cmap='gray')
        axes[3].imshow(combined, cmap='gray')

        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return utils.rescale_0_255(combined)

# def test_isolate_neurites(paths):
#     for path in paths:
#         img = plt.imread(path)
#         _isolate_neurites(img, show=False)
#     return

# def test_connect_broken_lines(paths):
#     for path in paths:
#         neurites_raw = plt.imread(path)
#         neurites = isolate(neurites_raw, show=True)
#         neurite_skeleton = _skeletonize(neurites)
#         connected = _connect_broken_lines(neurite_skeleton)

#         # # 4) close image to connect isolated pieces of neurite
#         # disk = skimage.morphology.disk(2)
#         # closed = skimage.morphology.closing(connected > 0, disk)

#         closed = _reconstruct(neurites, connected, show=True)

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

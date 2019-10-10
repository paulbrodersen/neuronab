import numpy as np
import matplotlib.pyplot as plt; # plt.ion(); plt.close('all')

import scipy.ndimage
import skimage.morphology

import cleaning
import utils
import neurites

global TITLE_FONT_SIZE
TITLE_FONT_SIZE = 'large'

def count(neurite_marker,
          primary_synaptic_marker,
          secondary_synaptic_marker=None,
          min_synapse_size=16,
          max_synapse_size=144,
          min_synapse_brightness=97.5,
          show=False,
          save=None):

    """
    Arguments:
    ----------
        neurite_marker: string or numpy.uint8 array
            path to grayscale image of neurite marker OR
            corresponding numpy array with values in the range (0-255)

        primary_synaptic_marker: string or numpy.uint8 array
            path to grayscale image of synaptic marker OR
            corresponding numpy array with values in the range (0-255)

        secondary_synaptic_marker: string or numpy.uint8 array (optional, default None)
            path to grayscale image of a secondary synaptic marker OR
            corresponding numpy array with values in the range (0-255)

        min_synapse_size: int
            minimum acceptable synapse sizes in pixels

        max_synapse_size: int
            maximum acceptable synapse sizes in pixels

        min_synapse_brightness: float in the range 0.-100., (default 97.5)
            image intensity threshold in percent above which objects
            in synaptic marker images are labelled as putative synapses

        show: bool (default False)
            if True, plots intermediate steps of image analysis

        save: str (default None)
    `      if not None (and show is True), figures will be saved under save+<integer>.pdf

    Returns:
    --------
        neurite_length: int
            total length of neurites in pixels

        primary_synaptic_marker_count: int
            number of synapses detected in image of primary synaptic marker

        secondary_synaptic_marker_count: int
            number of synapses detected in image of secondary synaptic marker;
            only returned if secondary_synaptic_marker is not None;

        dual_labelled: int
            number of synapses detected in images of both synaptic markers;
            only returned if secondary_synaptic_marker is not None;

    """
    plt.ion()
    plt.close('all')

    # manage input
    neurites_raw = utils.handle_grayscale_image_input(neurite_marker)
    primary_raw = utils.handle_grayscale_image_input(primary_synaptic_marker)

    if secondary_synaptic_marker != None:
        secondary_raw = utils.handle_grayscale_image_input(secondary_synaptic_marker)

    # isolate neurites and determine neurite length
    neurite_mask = neurites.isolate(neurites_raw, show)
    neurite_length = neurites.get_length(neurite_mask, show)

    # find synapse candidates and count
    primary = isolate(primary_raw, neurite_mask,
                      min_synapse_size, max_synapse_size, min_synapse_brightness, show)
    primary_count = _count_objects(primary)

    if secondary_synaptic_marker == None:

        if show == True:
            combined = utils.grayscale_to_rgb(utils.rescale_0_255(neurites_raw))
            combined[np.where(primary)] = np.array([255, 0, 0])

            # fig = plt.figure()
            # ax1 = fig.add_subplot(2,4,1)
            # ax1.imshow(neurites_raw, cmap='gray')
            # ax1.set_title('neurites')

            # ax2 = fig.add_subplot(2,4,2)
            # ax2.imshow(neurite_mask, cmap='gray')
            # ax2.set_title('neurite mask')

            # ax3 = fig.add_subplot(2,4,5)
            # ax3.imshow(primary_raw, cmap='gray')
            # ax3.set_title('primary synaptic marker')

            # ax4 = fig.add_subplot(2,4,6)
            # ax4.imshow(primary, cmap='gray')
            # ax4.set_title('isolated synapses')

            # ax5 = fig.add_subplot(1,2,2)
            # ax5.imshow(combined, cmap='gray')
            # ax5.set_title('neurites & isolated synapses')

            # for ax in [ax1, ax2, ax3, ax4, ax5]:
            #     ax.set_xticklabels([])
            #     ax.set_yticklabels([])
            # fig.tight_layout()

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(primary_raw, cmap='gray')
            ax1.set_title('primary synaptic marker')
            ax2.imshow(combined)
            ax2.set_title('neurites & isolated synapses')
            for ax in [ax1, ax2]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.tight_layout()

        if save != None:
            for ii in plt.get_fignums():
                plt.figure(ii)
                plt.savefig(save + '{}.pdf'.format(ii), dpi=300)

        return neurite_length, primary_count

    else:
        secondary = isolate(secondary_raw, neurite_mask,
                            min_synapse_size, max_synapse_size,
                            min_synapse_brightness, show)
        secondary_count = _count_objects(secondary)

        dual_labelled = _is_dual_labelled(primary, secondary, show=False)
        dual_labelled_count = _count_objects(dual_labelled)

        if show == True:
            combined = utils.grayscale_to_rgb(utils.rescale_0_255(neurites_raw))
            combined[np.where(primary)] = np.array([255, 0, 0])
            combined[np.where(secondary)] = np.array([0, 255, 0])

            # fig = plt.figure()
            # ax1 = fig.add_subplot(3,4,1)
            # ax1.imshow(neurites_raw, cmap='gray')
            # ax1.set_title('neurites')

            # ax2 = fig.add_subplot(3,4,2)
            # ax2.imshow(neurite_mask, cmap='gray')
            # ax2.set_title('neurite mask')

            # ax3 = fig.add_subplot(3,4,5)
            # ax3.imshow(primary_raw, cmap='gray')
            # ax3.set_title('primary synaptic marker')

            # ax4 = fig.add_subplot(3,4,6)
            # ax4.imshow(primary, cmap='gray')
            # ax4.set_title('isolated synapses')

            # ax5 = fig.add_subplot(3,4,9)
            # ax5.imshow(secondary_raw, cmap='gray')
            # ax5.set_title('secondary synaptic marker')

            # ax6 = fig.add_subplot(3,4,10)
            # ax6.imshow(secondary, cmap='gray')
            # ax6.set_title('isolated synapses')

            # ax7 = fig.add_subplot(1,2,2)
            # ax7.imshow(combined)
            # ax7.set_title('neurites & isolated synapses')

            # for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            #     ax.set_xticklabels([])
            #     ax.set_yticklabels([])
            # fig.tight_layout()

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(primary_raw + secondary_raw, cmap='gray')
            ax1.set_title('primary & secondary synaptic marker')

            ax2.imshow(combined)
            ax2.set_title('neurites & isolated synapses')

            for ax in [ax1, ax2]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.tight_layout()

        if save != None:
            for ii in plt.get_fignums():
                plt.figure(ii)
                plt.savefig(save + '{}.pdf'.format(ii), dpi=300)

        return neurite_length, primary_count, secondary_count, dual_labelled_count

def isolate(synaptic_marker,
            neurite_mask,
            min_synapse_size=16,
            max_synapse_size=144,
            min_synapse_brightness=97.5,
            show=False):

    """
    Arguments:
    ----------
        synaptic_marker: string or numpy.uint8 array
            path to grayscale image of synaptic marker, OR
            corresponding numpy array with values in the range (0-255)

        neurite_mask: string or numpy.bool array
            path to binary image indicating the presence of neurites, OR
            corresponding boolean numpy.ndarray

        min_synapse_size: int
            minimum acceptable synapse sizes in pixels

        max_synapse_size: int
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
    dilated = skimage.morphology.binary_dilation(neurite_mask, skimage.morphology.disk(2))
    synapse_mask = np.logical_and(cleaned, dilated)

    if show == True:
        cleaned = utils.rescale_0_255(cleaned) + 50 * neurite_mask

        images = [synapses_raw, thresholded, cleaned, synapse_mask]
        titles = ['Synaptic marker', 'Thresholded', 'Within size range', 'Within neurite mask']

        fig, axes = plt.subplots(2,2)
        # fig.suptitle('Synapse isolation', fontsize=TITLE_FONT_SIZE)
        for img, ax, title in zip(images, axes.ravel(), titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

    return synapse_mask

def _count_objects(binary_mask):
    label_objects, nb_labels = scipy.ndimage.label(binary_mask)
    count = label_objects.max()
    return count

def _is_dual_labelled(primary, secondary, show):
    dual_labelled = np.logical_and(primary, secondary)

    if show == True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(primary, cmap='gray')
        ax1.set_title('primary marker')
        ax2.imshow(secondary, cmap='gray')
        ax2.set_title('secondary marker')
        ax3.imshow(dual_labelled, cmap='gray')
        ax3.set_title('dual labelled')
        for ax in [ax1, ax2, ax3]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()

    return dual_labelled

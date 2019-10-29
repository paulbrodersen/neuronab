#!/usr/bin/env python
"""
Quantify anti-KDEL Ab stain in NeuN stained neurons and PV stained neurons.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import (
    disk,
    binary_dilation,
    binary_opening,
)
from neuronab import (
    utils,
    cleaning,
    somata,
)

if __name__ == '__main__':

    neun_path = 'test_images/neun.tif'
    pv_path   = 'test_images/pv.tif'
    kdel_path = 'test_images/kdel.tif'

    neun = utils.handle_grayscale_image_input(neun_path)
    kdel = utils.handle_grayscale_image_input(kdel_path)
    pv   = utils.handle_grayscale_image_input(pv_path)

    neun_mask = somata.get_mask(neun, intensity_threshold=50, show=True)
    # counts = somata.get_count(neun_mask, neun_path, threshold=0.05, show=True)
    counts = somata.get_count(neun_mask, threshold=0.05, show=True)
    print(f'Number of NeuN positive cells: {counts}')

    images = [kdel, neun, neun_mask, kdel * (neun_mask > 0)]
    titles = ['KDEL', 'NeuN', 'Soma mask', 'Masked KDEL']
    utils.plot_image_collection(images, titles, nrows=2, ncols=2)

    pv_mask = somata.get_mask(pv, intensity_threshold=95, show=True)
    # counts = somata.get_count(pv_mask, pv, threshold=0.1, show=True)
    counts = somata.get_count(pv_mask, threshold=0.1, show=True)
    print(f'Number of PV positive cells: {counts}')

    images = [kdel, pv, pv_mask, kdel * (pv_mask > 0)]
    titles = ['KDEL', 'PV', 'Soma mask', 'Masked KDEL']
    utils.plot_image_collection(images, titles, nrows=2, ncols=2)

    plt.show()
    input('Press any key to close figures...')
    plt.close('all')

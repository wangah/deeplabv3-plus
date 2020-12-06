import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cityscapesscripts.helpers import labels

import torch
from torch.utils.data import Dataset, DataLoader
import transforms




def get_palette():
    """
    Palette sequence used for the Cityscapes annotations.

    From PIL documentation:
    Palette sequence must contain 768 integer values, where each group
    of three values represent the red, green, and blue values for the
    corresponding pixel index.
    """
    palette = []
    trainId_to_color = {label.trainId: label.color for label in labels.labels}
    for i in range(19):
        palette.extend(trainId_to_color[i])
    zero_pads = [0] * (768 - len(palette))
    palette.extend(zero_pads)
    return palette


def color_mask(mask, palette):
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(palette)
    return mask

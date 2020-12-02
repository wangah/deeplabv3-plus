import os
import glob

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
from cityscapesscripts.helpers import labels


IMG_DIR = "./data/leftImg8bit"
MASK_DIR = "./data/gtFine"


def get_img_mask_paths(split, img_root=IMG_DIR, mask_root=MASK_DIR):
    assert split in ("train", "val")
    img_paths = sorted(
        glob.glob(os.path.join(img_root, split, "*", "*_leftImg8bit.png"))
    )
    mask_paths = sorted(
        glob.glob(os.path.join(mask_root, split, "*", "*_labelIds.png"))
    )
    return img_paths, mask_paths



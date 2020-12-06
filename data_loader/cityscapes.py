import os
import glob
from PIL import Image
from cityscapesscripts.helpers import labels

import torch
from torch.utils.data import Dataset


IMG_DIR = "../data/leftImg8bit"
MASK_DIR = "../data/gtFine"


def get_img_mask_paths(split, img_root=IMG_DIR, mask_root=MASK_DIR):
    assert split in ("train", "val")
    img_paths = sorted(
        glob.glob(os.path.join(img_root, split, "*", "*_leftImg8bit.png"))
    )
    mask_paths = sorted(
        glob.glob(os.path.join(mask_root, split, "*", "*_labelIds.png"))
    )
    return img_paths, mask_paths


class CityscapesDataset(Dataset):
    def __init__(self, split, img_root, mask_root, transform=None):
        self.split = split
        self.img_root = img_root
        self.mask_root = mask_root
        self.img_paths, self.mask_paths = get_img_mask_paths(split, img_root, mask_root)
        assert len(self.img_paths) == len(self.mask_paths) and len(self.img_paths) > 0

        self.transform = transform

        self.ignoreId = 19
        self.id_to_trainId = {
            label.id: (label.trainId if label.trainId not in (-1, 255) else 19)
            for label in labels.labels
        }
        self.trainId_to_id = {
            trainId: (label_id if trainId != 19 else 0)
            for label_id, trainId in self.id_to_trainId.items()
        }

    def convert_to_trainId(self, mask):
        """
        https://stackoverflow.com/questions/47171356
        """
        k = torch.tensor(list(self.id_to_trainId.keys()))
        v = torch.tensor(list(self.id_to_trainId.values()))
        sidx = k.argsort()

        ks = k[sidx]
        vs = v[sidx]
        return vs[torch.searchsorted(ks, mask)].to(torch.int64)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = self.convert_to_trainId(mask)
        return {"image": image, "mask": mask}

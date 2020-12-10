import os
import glob
from PIL import Image
import torch
from cityscapesscripts.helpers import labels
from torch.utils.data import Dataset, DataLoader
import utils.transforms


class CityscapesDataset(Dataset):
    def __init__(self, split, img_root, mask_root, transform=None):
        assert split in ("train", "val")
        self.split = split
        self.img_root = img_root
        self.mask_root = mask_root
        self.transform = transform

        self.img_paths, self.mask_paths = self.get_img_mask_paths()
        assert len(self.img_paths) == len(self.mask_paths) and len(self.img_paths) > 0

        self.num_classes = 20
        self.ignoreId = 19
        self.id_to_trainId = {
            label.id: (
                label.trainId if label.trainId not in (-1, 255) else self.ignoreId
            )
            for label in labels.labels
        }
        self.trainId_to_id = {
            trainId: (label_id if trainId != self.ignoreId else 0)
            for label_id, trainId in self.id_to_trainId.items()
        }
        self.name_to_train_Id = {
            (label.name if label.trainId not in (-1, 255) else "unlabeled"): (
                label.trainId if label.trainId not in (-1, 255) else self.ignoreId
            )
            for label in labels.labels
        }
        self.trainId_to_name = {
            trainId: name for name, trainId in self.name_to_train_Id.items()
        }

    def get_img_mask_paths(self):
        img_paths = sorted(
            glob.glob(os.path.join(self.img_root, self.split, "*", "*_leftImg8bit.png"))
        )
        mask_paths = sorted(
            glob.glob(os.path.join(self.mask_root, self.split, "*", "*_labelIds.png"))
        )
        return img_paths, mask_paths

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


class CityscapesDataLoader:
    def __init__(
        self,
        img_root,
        mask_root,
        batch_size=4,
    ):
        self.img_root = img_root
        self.mask_root = mask_root
        self.batch_size = batch_size
        self.train_transform = utils.transforms.Compose(
            [
                utils.transforms.RandomScaleCrop(
                    scale_min=0.5,
                    scale_max=2.0,
                    crop_size=512,
                    inference_size=(512, 1024),
                    fill=0,
                ),
                utils.transforms.RandomHorizontalFlip(flip_prob=0.5),
                utils.transforms.ToTensor(),
            ]
        )
        self.val_transform = utils.transforms.Compose(
            [
                utils.transforms.Resize((512, 1024)),
                utils.transforms.CenterCrop((512, 512)),
                utils.transforms.ToTensor(),
            ]
        )
        self.inference_transform = utils.transforms.Compose([utils.transforms.ToTensor()])

        self.train_set = CityscapesDataset(
            "train", img_root, mask_root, transform=self.train_transform
        )
        self.val_set = CityscapesDataset(
            "val", img_root, mask_root, transform=self.val_transform
        )
        self.inference_set = CityscapesDataset(
            "val", img_root, mask_root, transform=self.inference_transform
        )
        self.train_loader = DataLoader(
            dataset=self.train_set, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=self.val_set, batch_size=batch_size, shuffle=False
        )
        self.inference_loader = DataLoader(
            dataset=self.inference_set, batch_size=2, shuffle=True
        )

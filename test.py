import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader.cityscapes import CityscapesDataLoader
from model.metric import SegmentationMetrics
from model.deeplabv3plus import DeepLabv3Plus

from parse_config import ConfigParser


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = CityscapesDataLoader(
        config["data_loader"]["args"]["img_dir"],
        config["data_loader"]["args"]["mask_dir"],
        config["data_loader"]["args"]["batch_size"],
        config["data_loader"]["args"]["num_workers"],
    )
    inference_loader = data_loader.inference_loader
    trainId_to_name = data_loader.val_set.trainId_to_name

    # build model architecture
    num_classes = config["arch"]["args"]["num_classes"]
    model = DeepLabv3Plus(num_classes)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # get function handles of loss and metrics
    total_loss = 0.0
    ignore_idx = config["ignore_idx"]
    criterion = nn.CrossEntropyLoss(ignore_idx)
    metrics = SegmentationMetrics(num_classes, ignore_idx)

    with torch.no_grad():
        for _, sample in enumerate(tqdm(inference_loader)):
            images = sample["image"].to(device)
            masks = sample["mask"].to(device)

            pred = model(images)
            loss = criterion(pred, masks)
            total_loss += loss.item()

            pred_cls = torch.argmax(pred, dim=1)
            metrics.update(pred_cls, masks)

    ious, mIoU = metrics.iou()
    logger.info({"total_loss": total_loss, "mIoU": mIoU})
    logger.info({trainId_to_name[i] + "_iou": iou for i, iou in enumerate(ious)})


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="DeepLabv3Plus")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)

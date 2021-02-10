import argparse
import numpy as np
import torch
import torch.nn as nn
from data_loader.cityscapes import CityscapesDataLoader
from model.deeplabv3plus import DeepLabv3Plus
from model.metric import SegmentationMetrics
from parse_config import ConfigParser
from torch_poly_lr_decay import PolynomialLRDecay
from torchsummary import summary
from trainer.trainer import Trainer
from utils.utils import prepare_device


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    data_loader = CityscapesDataLoader(
        config["data_loader"]["args"]["data_root"],
        config["data_loader"]["args"]["train_extra"],
        config["data_loader"]["args"]["batch_size"],
        config["data_loader"]["args"]["num_workers"],
    )

    num_classes = config["arch"]["args"]["num_classes"]
    model = DeepLabv3Plus(num_classes=num_classes)
    logger.info(
        summary(
            model,
            (3, 1024, 2048),
            col_names=("kernel_size", "output_size", "num_params"),
            depth=5,
            verbose=0,
        )
    )

    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    ignore_index = config["loss"]["args"]["ignore_index"]
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    metrics = SegmentationMetrics(num_classes, ignore_index)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["optimizer"]["args"]["lr"],
        momentum=config["optimizer"]["args"]["momentum"],
        weight_decay=config["optimizer"]["args"]["weight_decay"],
    )
    lr_scheduler = PolynomialLRDecay(
        optimizer,
        max_decay_steps=config["lr_scheduler"]["args"]["max_decay_steps"],
        end_learning_rate=config["lr_scheduler"]["args"]["end_learning_rate"],
        power=config["lr_scheduler"]["args"]["power"],
    )

    trainer = Trainer(
        config=config,
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        train_loader=data_loader.train_loader,
        val_loader=data_loader.val_loader,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="DeepLabv3Plus")
    args.add_argument(
        "-c", "--config", default="./config.json", type=str, help="config file path"
    )
    args.add_argument(
        "-d", "--device", default=None, type=str, help="indices of GPUs to enable"
    )

    config = ConfigParser.from_args(args)
    main(config)

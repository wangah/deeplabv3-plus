import math
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from model.metric import SegmentationMetrics


class Trainer:
    def __init__(
        self,
        iterations,
        model,
        criterion,
        accumulate_grad_batches,
        optimizer,
        lr_scheduler,
        device,
        num_classes,
        ignore_idx,
        train_loader,
        val_loader,
        save_freq,
        ckpt_path,
        tensorboard_dir,
    ):
        self.iterations = iterations
        self.model = model
        self.criterion = criterion
        self.accumulate_grad_batches = accumulate_grad_batches
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_freq = save_freq
        self.ckpt_path = ckpt_path
        self.tensorboard_dir = tensorboard_dir
        self.writer = SummaryWriter(tensorboard_dir)

        self.iters_per_epoch = math.floor(
            len(self.train_loader) / self.accumulate_grad_batches
        )
        self.epochs = math.ceil(self.iterations / self.iters_per_epoch)
        self.start_epoch = 1

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.ckpt_path,
        )

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        accumulated_loss = 0

        self.optimizer.zero_grad()
        for i, sample in enumerate(tqdm(self.train_loader)):
            images = sample["image"].to(self.device)
            masks = sample["mask"].to(self.device)

            pred = self.model(images)
            loss = self.criterion(pred, masks)
            loss = loss / self.accumulate_grad_batches
            loss.backward()
            accumulated_loss += loss.item()

            if (i + 1) % self.accumulate_grad_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.writer.add_scalar(
                    "train_iter_loss",
                    accumulated_loss,
                    ((i + 1) / self.accumulate_grad_batches)
                    + (self.iters_per_epoch * (epoch - 1)),
                )
                total_loss += accumulated_loss
                accumulated_loss = 0

        avg_loss = total_loss / self.iters_per_epoch
        self.writer.add_scalar("train_epoch_avg_loss", avg_loss, epoch)
        return avg_loss

    def _valid_epoch(self, epoch):
        self.model.eval()

        total_loss = 0
        n_batches = len(self.val_loader)
        metrics = SegmentationMetrics(
            num_classes=self.num_classes, ignore_idx=self.ignore_idx
        )
        with torch.no_grad():
            for _, sample in enumerate(tqdm(self.val_loader)):
                images = sample["image"].to(self.device)
                masks = sample["mask"].to(self.device)

                pred = self.model(images)
                loss = self.criterion(pred, masks)
                total_loss += loss.item()

                pred_cls = torch.argmax(pred, dim=1)
                metrics.update(pred_cls, masks)

        self.model.train()

        avg_loss = total_loss / n_batches
        ious, mIoU = metrics.iou()
        self.writer.add_scalar("val_epoch_avg_loss", avg_loss, epoch)
        self.writer.add_scalar("val_epoch_mIoU", mIoU, epoch)
        return avg_loss, ious, mIoU

    def train(self):
        best_mIoU = 0
        best_epoch = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, ious, mIoU = self._valid_epoch(epoch)
            print(f"Epoch {epoch}: val_loss {val_loss:.4f} | mIoU {mIoU:.4f}")

            if mIoU > best_mIoU:
                best_mIoU = mIoU
                best_epoch = epoch
                self.save_checkpoint(epoch)

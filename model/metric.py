import torch
import numpy as np
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    def __init__(self, num_classes=20, ignore_idx=19):
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.labels = np.arange(num_classes)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """
        pred (tensor), shape [B, C, H, W]
        target (tensor), shape[B, C, H, W]

        https://discuss.pytorch.org/t/confusion-matrix/21026
        """
        with torch.no_grad():
            y_pred = pred.view(-1).cpu().detach().numpy()
            y = target.view(-1).cpu().detach().numpy()
            update_counts = confusion_matrix(
                y_true=y, y_pred=y_pred, labels=self.labels
            )
            self.confusion_matrix += update_counts

    def iou(self):
        """
        IoU = TP / (TP + FP + FN)
        """
        intersection = np.diag(self.confusion_matrix)
        pred_counts = np.sum(self.confusion_matrix, axis=0)
        target_counts = np.sum(self.confusion_matrix, axis=1)
        union = pred_counts + target_counts - intersection

        ious = intersection / union
        ious_no_ignore = np.delete(ious, self.ignore_idx)
        return ious, np.mean(ious_no_ignore)

    def pixelwise_accuracy(self):
        tp = np.sum(np.diag(self.confusion_matrix))
        total = np.sum(self.confusion_matrix)
        return tp / total

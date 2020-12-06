import torch
import numpy as np


def iou(output, target, num_classes=20, ignore_idx=19):
    """
    Credit:
    https://stackoverflow.com/questions/48260415
    """
    with torch.no_grad():
        ious = []
        # pred = torch.argmax(output, dim=1)  # TODO check this
        pred = output.view(-1)
        target = target.view(-1)
        print(pred.shape)
        print(target.shape)

        for k in range(num_classes):
            if k == ignore_idx:
                break
            pred_inds = pred == k
            target_inds = target == k
            intersection = (pred_inds[target_inds]).long().sum().data.cpu()
            union = (
                pred_inds.long().sum().data.cpu()
                + target_inds.long().sum().data.cpu()
                - intersection
            )
            if union == 0:
                ious.append(float("nan"))
            else:
                ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious), np.mean(ious)


# def accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    """Soft IoU on probabilities (not logits), averaged per sample."""
    def __init__(self, batch=None, smooth=1e-6):
        super(SoftLoULoss, self).__init__()
        if smooth <= 0:
            raise ValueError('smooth must be positive')
        self.smooth = smooth

    def forward(self, prob, target):
        target = target.to(dtype=prob.dtype)
        dims = (1, 2, 3)
        intersection = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims) - intersection
        return (1 - (intersection + self.smooth) /
                (union + self.smooth)).mean()


class EdgeLoss(nn.Module):
    """BCE supplies gradients on empty masks; IoU supervises overlap."""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.iou = SoftLoULoss()

    def components(self, logits, target):
        target = target.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        iou = self.iou(torch.sigmoid(logits), target)
        return bce, iou

    def forward(self, logits, target):
        bce, iou = self.components(logits, target)
        return bce + iou

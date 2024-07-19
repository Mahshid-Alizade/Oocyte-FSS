import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        logits: predicted outputs from the model (batch_size, num_classes)
        labels: ground truth labels (batch_size)
        """
        # Ensure labels are in the correct shape (batch_size)
        if labels.ndim == 2:
            labels = labels.squeeze(dim=1)

        # Compute the cross-entropy loss
        loss = self.cross_entropy(logits, labels)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        probs = torch.sigmoid(logits)
        num = true.size(0)
        m1 = probs.view(num, -1)
        m2 = true.view(num, -1)
        intersection = (m1 * m2).sum()
        return 1 - ((2. * intersection + self.smooth) / (m1.sum() + m2.sum() + self.smooth))
    
# criterion = DiceLoss()

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        probs = torch.sigmoid(logits)
        num = true.size(0)
        m1 = probs.view(num, -1)
        m2 = true.view(num, -1)
        intersection = (m1 * m2).sum()
        union = m1.sum() + m2.sum() - intersection
        return 1 - ((intersection + self.smooth) / (union + self.smooth))

# criterion = JaccardLoss()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, true):
        probs = torch.sigmoid(logits)
        num = true.size(0)
        m1 = probs.view(num, -1)
        m2 = true.view(num, -1)
        intersection = (m1 * m2).sum()
        fp = ((1 - m2) * m1).sum()
        fn = ((1 - m1) * m2).sum()
        return 1 - ((intersection + self.smooth) / (intersection + self.alpha * fp + self.beta * fn + self.smooth))

# criterion = TverskyLoss(alpha=0.5, beta=0.5)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, true):
        ce = self.ce_loss(logits, true)
        dice = self.dice_loss(logits, true)
        return self.alpha * ce + (1 - self.alpha) * dice

# criterion = CombinedLoss(alpha=0.5)




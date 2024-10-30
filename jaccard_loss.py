import torch
import torch.nn as nn

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss,self).__init__()
        self.smooth = smooth

    def forward(self,preds, targets):

        preds = torch.sigmoid(preds)

        preds = preds.view(-1)

        targets = targets.view(-1)

        intersection = (preds*targets).sum()
        union = preds.sum() + targets.sum() - intersection

        return 1- ((intersection+self.smooth) / (union + self.smooth))
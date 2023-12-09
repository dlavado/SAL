import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.functional import accuracy



class Classifier_OutLayer(nn.Module):
    
    def __init__(self, in_channels:int, num_classes:int):
        """

        """
        super().__init__()
        #self.l1 = torch.nn.Linear(28*28, self.hparams.in_channels)
       
        self.l2 = torch.nn.Linear(in_channels, num_classes)
 
    def forward(self, x):
        #x = x.view(x.size(dim=0), -1) # flatten
        x = torch.flatten(x, start_dim=1)
        x = self.l2(x)
        return x
 
    def _compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)
 
    def _accuracy(self, logits, y):
       return accuracy(logits, y)

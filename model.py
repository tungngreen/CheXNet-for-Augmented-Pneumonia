import os
import torch
import torchvision
import torch.nn as nn

import torchvision.models as models

class CheXNet(nn.Module):
    def __init__(self, n_class, dense_net, is_trained):
        super(CheXNet, self).__init__()
        if dense_net == '121':
            self.densenet121 = models.densenet121(pretrained = is_trained)
        elif dense_net == '161':
            self.densenet161 = models.densenet161(pretrained = is_trained)
        elif dense_net == '169':
            self.densenet169 = models.densenet169(pretrained = is_trained)
        else:
            self.densenet201 = models.densenet201(pretrained = is_trained)
        
        in_features = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features, int(n_class)),
            nn.Sigmoid()
        )



    def forward(self, x):
        return self.densenet121(x)

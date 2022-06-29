import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    """
    An example pytorch model for classifying iris flower
    """
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 102))
        self.model.load_state_dict(torch.load('flower_classification.pt'))

    def forward(self, x):
        return self.model(x)
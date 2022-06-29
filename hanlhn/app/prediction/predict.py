from fastapi import File
import timm
import torch
from PIL import Image
from torchvision import transforms
from models.model import Model
import numpy as np
import pandas as pd
import torch.nn as nn

obj = pd.read_pickle('index2name.pkl')
eng_name = list(obj)


class Prediction:
    def __init__(self, device) -> None:
        self.model = timm.create_model("resnet50", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 102))
        self.model.to(device)
        self.model.load_state_dict(torch.load('models/flower_classification.pt', map_location = device))

    def preprocess(self, img):
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        # get normalized image
        img_normalized = transform_norm(img).float()
        img_normalized = img_normalized.unsqueeze(0)
        return img_normalized

    def predict(self, input) -> str:
        # process data
        preprocessed_input = self.preprocess(input)
        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(preprocessed_input)
            index = y_pred.data.cpu().numpy().argmax()
            name_pred = eng_name[index]

        return name_pred
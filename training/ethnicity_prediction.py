import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
import os
import numpy as np

class EthnicityPredictor(nn.Module):
    """Ethnicity predictor trained on FairFace, see https://github.com/dchen236/FairFace """
    def __init__(self, version="resnet34_7", device="cuda"):
        super(EthnicityPredictor, self).__init__()
        assert version in ["resnet34_7", "resnet34_4"], "Invalid version"
        if version == "resnet34_7":
            self.model_path = 'pretrained_models/fairface_classifier/res34_fair_align_multi_7_20190809.pt'
        elif version == "resnet34_4":
            self.model_path = 'pretrained_models/fairface_classifier/res34_fair_align_multi_4_20190809.pt'
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(device)
        self.model.eval()
        self.trans = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.version = version

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.trans(x)
        outputs = self.model(x)
        if self.version == "resnet34_7":
            race_outputs = outputs[:, :7]
        elif self.version == "resnet34_4":
            race_outputs = outputs[:, :4]
        race_score = torch.nn.functional.softmax(race_outputs, dim=1)
        return race_score
    
    def predict(self, x):
        with torch.no_grad():
            race_score = self.forward(x)
            race_pred = torch.argmax(race_score, axis=1)
            race_pred_str = ["None"] * len(race_pred)
            for j in range(len(race_pred)):
                race_pred_str[j] = code_to_eth(race_pred[j], version=self.version)
            return race_pred, race_pred_str

# Utility functions to map from int-code to the ethnicity and vice versa 
def code_to_eth(code, version="resnet34_7"):
    if version=="resnet34_7":
        if code==0:
            return "White"
        elif code==1:
            return "Black"
        elif code==2:
            return "Latino_Hispanic"
        elif code==3:
            return "East Asian"
        elif code==4:
            return "Southeast Asian"
        elif code==5:
            return "Indian"
        elif code==6:
            return "Middle Eastern"
    elif version=="resnet34_4":
        if code==0:
            return "White"
        elif code==1:
            return "Black"
        elif code==2:
            return "Asian"
        elif code==3:
            return "Indian"

def eth_to_code(eth, version="resnet34_7"):
    if version=="resnet34_7":
        if eth=="White":
            return 0
        elif eth=="Black":
            return 1
        elif eth=="Latino_Hispanic":
            return 2
        elif eth=="East Asian":
            return 3
        elif eth=="Southeast Asian":
            return 4
        elif eth=="Indian":
            return 5
        elif eth=="Middle Eastern":
            return 6
    elif version=="resnet34_4":
        if eth=="White":
            return 0
        elif eth=="Black":
            return 1
        elif eth=="Asian":
            return 2
        elif eth=="Indian":
            return 3
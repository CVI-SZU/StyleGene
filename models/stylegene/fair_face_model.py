import torch
import numpy as np
from PIL import Image
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from configs import path_ckpt_fairface

# code adapted from https://github.com/dchen236/FairFace

def init_fair_model(device, path_ckpt=None):
    if path_ckpt is None:
        path_ckpt = path_ckpt_fairface
    model_fair_7 = torchvision.models.resnet34(pretrained=False)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(
        torch.load(path_ckpt))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()
    return model_fair_7


def predict_race(model_fair_7, path_img, device):
    if type(path_img) == str:
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(path_img)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
    elif type(path_img) == torch.Tensor:
        trans = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = F.interpolate(path_img, (224, 224))
        image = image * 0.5 + 0.5
        image = trans(image)
        image = image.view(1, 3, 224, 224)

    image = image.to(device)

    outputs = model_fair_7(image)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.squeeze(outputs)

    race_outputs = outputs[:7]
    gender_outputs = outputs[7:9]
    age_outputs = outputs[9:18]

    race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
    age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

    race_pred = np.argmax(race_score)
    gender_pred = np.argmax(gender_score)
    age_pred = np.argmax(age_score)
    race_label = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
    return race_label[race_pred], race_pred, gender_pred, age_pred

import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    # Use MobileNetV2 for lightweight and fast inference
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze earlier layers to speed up training if dataset is small
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last few layers to finetune
    for param in model.features[-2:].parameters():
        param.requires_grad = True
        
    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    
    return model

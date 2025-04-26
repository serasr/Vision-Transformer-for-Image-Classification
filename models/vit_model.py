import torch
from models.modeling import VisionTransformer, CONFIGS
import numpy as np

def load_vit_model(pretrained_path, num_classes=2):
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, img_size=224, zero_head=True, num_classes=num_classes)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(config.hidden_size, num_classes)
    )
    model.load_from(np.load(pretrained_path))
    return model

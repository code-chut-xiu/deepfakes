import numpy as np
import pandas as pd
import os
from PIL import Image, ImageChops, ImageEnhance
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Precision, Recall
from torchvision import transforms

def image_to_ela(image_path: str, quality: int = 90, scale: int = 15):
    if image_path.endswith('png') or image_path.endswith('jpg') or image_path.endswith('jpeg'):
        try: 
            original = Image.open(image_path).convert('RGB')
        
            temp_path = "temp.jpg"
            original.save(temp_path, 'JPEG', quality = quality)
        
            recompressed = Image.open(temp_path).convert('RGB')
            diff = ImageChops.difference(original, recompressed)
        
            extrema = diff.getextrema()
            max_diff = max([ ex[1] for ex in extrema]) or 1
            scale_factor = 255.0 / max_diff if max_diff > 0 else scale

            return ImageEnhance.Brightness(diff).enhance(scale_factor)
        except Exception as ex:
            print(f"Failed to convert {image_path} to ELA: {str(ex)}")
    else:
        print(f"Only PNG, JPG or JPEG image format is supported.")


class Model(nn.Module):
    def __init__(self, in_features, h1, h2, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x
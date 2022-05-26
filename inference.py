#%%
import torch # For Torch Deep Learning framework
import itertools # Extract subset of generator
import glob # getting paths from directory
from tqdm import tqdm # For progress bar
import numpy as np # Base library for computations
import matplotlib.pyplot as plt # For visualization
import torchvision # Pretrained models/Transforms
from sklearn.model_selection import train_test_split # Splitting data
import os # OS operations
import cv2 # For image processing operations
import torch.nn as nn 
from utils import DefaultConfig, MetricMonitor, calculate_accuracy, get_transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import reduce
from collections import defaultdict

# CONFIGURATION
class DefaultConfig:
  seed = 42
  labels = ['cat', 'dog']
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  batch_size = 16
  img_size = (224, 224)
  epochs = 3

def set_seed(SEED):
  os.environ['PYTHONHASHSEED'] = str(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  print(f"SEED {SEED} set!")

set_seed(DefaultConfig.seed)

paths = glob.glob("../kagglecatsanddogs_3367a/**/*.jpg", recursive=True)
_, paths = train_test_split(paths, test_size=0.25, random_state=DefaultConfig.seed)

with open('test_paths.txt', 'w') as f:
    for item in paths:
        f.write("%s\n" % item)

class CatsDogsModel(torch.nn.Module):
    def __init__(self, model):
        super(CatsDogsModel, self).__init__()
        self.model = model
    def forward(self, x, training=False):
        transforms = self.transforms(training)
        x = transforms(x)
        x = self.model(x)
        return x
    def transforms(self, training=False):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        custom_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: x / 255.0),
        torchvision.transforms.Lambda(lambda x: x.permute(2, 0, 1)),
        torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
        ])
        if training:
            return torchvision.transforms.Compose([
                custom_to_tensor,
                torchvision.transforms.Normalize(mean, std),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.GaussianBlur(5)])
        else:
            return torchvision.transforms.Compose([
                custom_to_tensor,
                torchvision.transforms.Normalize(mean, std)])

model_url = "models/mobilenet_v2_cats_dogs_16.pt"
model = torch.load(model_url, map_location=DefaultConfig.device)
model.eval()
inference_model = CatsDogsModel(model)

images = next(iter(test_data))

plt.figure(figsize=(10, 10))
for i, image in enumerate(images):
    plt.subplot(4, 4, i + 1)
    out_ = inference_model(image)
    output = torch.sigmoid(out_) >= 0.5
    plt.imshow(image)
    title = "Dog" if output.item() else "Cat"
    plt.title(title)
    plt.axis("off")
plt.show()

torch.onnx.export(inference_model, torch.rand(224, 224, 3), "model.onnx", verbose=True)

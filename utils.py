import os
import numpy as np
import random
import torch
from collections import defaultdict
import albumentations as A
class DefaultConfig:
  seed = 42
  labels = ['cat', 'dog']
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  batch_size = 16
  img_size = (224, 224)
  epochs = 10

def set_seed(SEED):
  os.environ['PYTHONHASHSEED'] = str(SEED)
  np.random.seed(SEED)
  random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  print(f"SEED {SEED} set!")

set_seed(DefaultConfig.seed)

def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()

def get_transforms(training=True):
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  if training:
    return A.Compose([
    A.transforms.Normalize(mean = mean, std = std, always_apply=True, p=1.0),
    A.transforms.CoarseDropout(max_holes=30, max_height=10, max_width=10, fill_value=64),
    A.transforms.Flip(),
    ])
  else:
    return A.Compose([
    A.transforms.Normalize(mean=mean, std=std, always_apply=True, p=1.0),
    ])

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

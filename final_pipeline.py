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

train_paths, test_paths = train_test_split(
    paths, test_size=0.25, random_state=DefaultConfig.seed)

print(f"Train images: {len(train_paths)} Test images: {len(test_paths)}")

# Data loading pipeline
def transforms(training=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if training:
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.GaussianBlur(5)
        ])
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])

class CatsDogsDataset(Dataset):
  def __init__(self, image_path, img_size, transforms=None, training=False):
    self.image_path = image_path
    self.training = training
    self.img_size = img_size
    self.transforms = transforms
  def __len__(self):
    return len(self.image_path)
  def __getitem__(self, idx):
    path = self.image_path[idx]
    image = self.read_image(path)
    if self.training:
      label = (path.split("\\")[-2])
      label = 0 if label == 'Cat' else 1
      return image, label
    else:
      return image
  def read_image(self, path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, self.img_size)
    return self.transforms(image)

def get_data(
    train=train_paths, test=test_paths,
    batch_size=DefaultConfig.batch_size, num_workers=0):
  train_data = CatsDogsDataset(train, DefaultConfig.img_size, transforms(True), True)
  test_data = CatsDogsDataset(test,  DefaultConfig.img_size,  transforms(False), False)
  train_loader = DataLoader(train_data, batch_size=batch_size,
                            num_workers=num_workers, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)
  return train_loader, test_loader

train_data, test_data = get_data()
images, labels = next(iter(train_data))

def inverse_norm(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = ((np.asarray(image) * np.array(std) * 255) +
             (np.array(mean) * 255)).astype(np.uint8)
    return image

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(
    inverse_norm(image.permute(1, 2, 0).numpy())
    )
    plt.title(DefaultConfig.labels[label])
    plt.axis('off')
plt.show()

# Building model
model = torchvision.models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
fc_in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(fc_in_features, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

def count_params(model):
    trainable_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    print("Trainable parameters in model : {} ({:.2f} %)\n"
    "Total parameters: {}".format(
      trainable_params,
      trainable_params / total_params * 100,
      total_params))

count_params(model)

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

criterion = torch.nn.BCEWithLogitsLoss().to(DefaultConfig.device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
model = model.to(DefaultConfig.device)



def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()

def train(train_loader, model, criterion, optimizer, epoch):
  metric_monitor = MetricMonitor()
  model.train()
  stream = tqdm(train_loader)
  for index, (images, targets) in enumerate(stream, start=1):
    images = images.to(DefaultConfig.device , non_blocking=True)
    targets = targets.to(DefaultConfig.device , non_blocking=True).float().view(-1, 1)
    output = model(images)
    loss = criterion(output, targets)
    accuracy = calculate_accuracy(output, targets)
    metric_monitor.update("Loss", loss.item())
    metric_monitor.update("Accuracy", accuracy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    stream.set_description(
        "Epoch: {epoch}. Train. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
    )


# Let's get the first 32 samples of train data for training
sample_data = list(itertools.islice(train_data, DefaultConfig.batch_size*1))

for epoch in range(1, DefaultConfig.epochs + 1):
  train(train_data, model, criterion, optimizer, epoch)
  if epoch % 10 == 0 and epoch != 0:
    torch.save(model, f"model_cats_dogs_{epoch}.pt")

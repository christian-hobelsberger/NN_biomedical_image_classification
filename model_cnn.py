import os
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator
from PIL import Image

data_flag = 'bloodmnist'
download = True
BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

data_blood_train = torch.load("data/data_blood_train.pt")
data_blood_test = torch.load("data/data_blood_test.pt")
data_blood_val = torch.load("data/data_blood_val.pt")

# Encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=data_blood_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=data_blood_val, batch_size=BATCH_SIZE, shuffle=False)
val_loader = data.DataLoader(dataset=data_blood_test, batch_size=BATCH_SIZE, shuffle=False)
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

# Preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load the BloodMNIST dataset
data_blood_train = DataClass(split='train', transform=data_transform, download=download)
data_blood_val = DataClass(split='val', transform=data_transform, download=download)
data_blood_test = DataClass(split='test', transform=data_transform, download=download)

# Encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=data_blood_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=data_blood_val, batch_size=BATCH_SIZE, shuffle=False)
val_loader = data.DataLoader(dataset=data_blood_test, batch_size=BATCH_SIZE, shuffle=False)

torch.save(data_blood_train, "data/data_blood_train.pt")
torch.save(data_blood_test, "data/data_blood_test.pt")
torch.save(data_blood_val, "data/data_blood_val.pt")
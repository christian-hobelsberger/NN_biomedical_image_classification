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

# Visualization
montage_train_1 = data_blood_train.montage(length=1)
montage_train_20 = data_blood_train.montage(length=20)

plots_folder = 'plots/'
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

montage_train_1.save(os.path.join(plots_folder, 'montage_train_1.png'))
montage_train_20.save(os.path.join(plots_folder, 'montage_train_20.png'))
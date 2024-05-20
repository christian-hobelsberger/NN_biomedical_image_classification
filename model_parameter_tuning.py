import os
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToTensor
import medmnist
from medmnist import INFO
from PIL import Image
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import numpy
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split


class convNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Conv2d(3, 64, 2),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          nn.Dropout(p=0.25),
          nn.Conv2d(64, 64, 2),
          nn.ReLU(),
          nn.Dropout(p=0.25),
          nn.Flatten(),
          nn.Linear(64*12*12, 64),
          nn.Dropout(p=0.25),
          nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.layers(x)




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

# Combine train and validation datasets
dataset = ConcatDataset([data_blood_train, data_blood_val])


def split_dataset(dataset):
    features = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]
    
    features = torch.stack(features)
    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.long).squeeze()
    return features, labels
    

# Split the combined dataset
X, y = split_dataset(dataset)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Features shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")

# Define the model
model = NeuralNetClassifier(
    convNet,
    criterion=nn.CrossEntropyLoss,
    max_epochs=10,
    optimizer=torch.optim.Adam,   
    verbose=1,
)

# Define hyperparameter grid
param_grid = {
    'batch_size': [16, 32],
    'max_epochs': [10, 50],
    'lr': [0.001, 0.01, 0.1, 0.2, 0.3],
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


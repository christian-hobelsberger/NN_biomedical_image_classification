import os
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import ConcatDataset
import medmnist
from medmnist import INFO
from PIL import Image
from sklearn.model_selection import KFold

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class convNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        

        # implement the cnn 

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
dataset = ConcatDataset([data_blood_train,data_blood_test, data_blood_val])


# Encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=data_blood_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=data_blood_val, batch_size=BATCH_SIZE, shuffle=False)
val_loader = data.DataLoader(dataset=data_blood_test, batch_size=BATCH_SIZE, shuffle=False)


k_folds = 10
num_epochs = 1
loss_function = nn.CrossEntropyLoss() 

# k-fold cross validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# fold results
results = {}

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # select 
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=BATCH_SIZE, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=BATCH_SIZE, sampler=test_subsampler)
    
    print(len(dataset))


    # init neural network
    model = convNet()
    model.apply(reset_weights)


    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      current_loss = 0.0

      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        
        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
       
       
            
    
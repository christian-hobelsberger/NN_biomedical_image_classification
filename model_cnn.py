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
        super(convNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.layers(x)

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

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
dataset = ConcatDataset([data_blood_train, data_blood_test, data_blood_val])

k_folds = 10
num_epochs = 50
learning_rate = 0.001
loss_function = nn.CrossEntropyLoss()

# k-fold cross validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# fold results
results = {}

# fixed number seed
torch.manual_seed(42)

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

    # Initialize models
    cnn_model = convNet()
    ffnn_model = FFNN(input_size=3 * 28 * 28, hidden_size=128, num_classes=n_classes)

    models = {'CNN': cnn_model, 'FFNN': ffnn_model}

    for model_name, model in models.items():
        print(f'Training {model_name}...')
        model.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):
            print(f'Starting epoch {epoch+1} for {model_name}')
            current_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                if targets.dim() > 1:  
                    targets = targets.squeeze()
                targets = targets.long()  
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

            print(f'Epoch {epoch+1} for {model_name}, Loss: {current_loss/len(trainloader)}')

        print(f'Training for {model_name} completed!')

        # Evaluation on test set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, targets = data
                if targets.dim() > 1:  
                    targets = targets.squeeze()
                targets = targets.long()  
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the {model_name} on the test set: {accuracy}%')
        results[fold] = accuracy

print('--------------------------------')
print('K-FOLD CROSS VALIDATION RESULTS')
print('--------------------------------')
sum_accuracy = 0
for key, value in results.items():
    print(f'Fold {key}: {value}%')
    sum_accuracy += value
print(f'Average accuracy: {sum_accuracy/len(results.items())}%')

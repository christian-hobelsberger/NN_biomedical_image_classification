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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

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

        true_labels = []
        pred_labels = []

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

                true_labels.extend(targets.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Accuracy of the {model_name} on the test set: {accuracy}%')
        results[(fold, model_name)] = accuracy

        if model_name == 'CNN':
            cnn_true_labels = true_labels
            cnn_pred_labels = pred_labels
        elif model_name == 'FFNN':
            ffnn_true_labels = true_labels
            ffnn_pred_labels = pred_labels

print('--------------------------------')
print('K-FOLD CROSS VALIDATION RESULTS')
print('--------------------------------')
sum_accuracy_cnn = 0
sum_accuracy_ffnn = 0
for key, value in results.items():
    fold, model_name = key
    print(f'Fold {fold} for {model_name}: {value}%')
    if model_name == 'CNN':
        sum_accuracy_cnn += value
    elif model_name == 'FFNN':
        sum_accuracy_ffnn += value

print(f'Average accuracy for CNN: {sum_accuracy_cnn/k_folds}%')
print(f'Average accuracy for FFNN: {sum_accuracy_ffnn/k_folds}%')

# Create and plot the confusion matrix for CNN
cm_cnn = confusion_matrix(cnn_true_labels, cnn_pred_labels)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=[str(i) for i in range(n_classes)])
disp_cnn.plot(cmap=plt.cm.Blues)
plt.title('CNN Confusion Matrix')
plt.show()

# Create and plot the confusion matrix for FFNN
cm_ffnn = confusion_matrix(ffnn_true_labels, ffnn_pred_labels)
disp_ffnn = ConfusionMatrixDisplay(confusion_matrix=cm_ffnn, display_labels=[str(i) for i in range(n_classes)])
disp_ffnn.plot(cmap=plt.cm.Blues)
plt.title('Baseline Confusion Matrix')
plt.show()

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
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns
import numpy as np

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

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*28*28, 8)
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
dataset = ConcatDataset([data_blood_train, data_blood_test, data_blood_val])

k_folds = 5
num_epochs = 5
learning_rate = 0.001
loss_function = nn.CrossEntropyLoss()

# k-fold cross validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# fixed number seed
torch.manual_seed(42)

# Track losses and accuracies
results = {
    'CNN': {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    },
    'Baseline': {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
}

'''
Training phase
'''
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    # Select
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    validation_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    validationloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_subsampler)

    # Initialize models
    cnn_model = convNet()
    Baseline_model = Baseline()
    models = {'CNN': cnn_model, 'Baseline': Baseline_model}

    for model_name, model in models.items():
        print(f'Training {model_name}...')
        model.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_losses_epochs = []
        val_losses_epochs = []
        train_accuracy_epochs = []
        val_accuracy_epochs = []

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):
            print(f'Starting epoch {epoch+1} for {model_name}')

            # Training phase
            model.train()
            total_loss = 0.0
            total_samples_train = 0
            total_correct_train = 0

            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                optimizer.zero_grad()

                outputs = model(inputs)
                outputs = outputs.squeeze()
                targets = targets.squeeze().type(torch.long)

                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total_loss += loss.item()
                total_samples_train += targets.size(0)
                total_correct_train += (predicted == targets).sum().item()

            train_loss = total_loss / len(trainloader)
            train_losses_epochs.append(train_loss)
            train_accuracy = total_correct_train / total_samples_train
            train_accuracy_epochs.append(train_accuracy)

            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

            # Validation phase
            model.eval()
            total_loss_val = 0.0
            total_samples_val = 0
            total_correct_val = 0

            with torch.no_grad():
                for inputs_val, targets_val in validationloader:
                    outputs_val = model(inputs_val)
                    outputs_val = outputs_val.squeeze()
                    targets_val = targets_val.squeeze().type(torch.long)

                    loss_val = loss_function(outputs_val, targets_val)
                    _, predicted_val = torch.max(outputs_val, 1)

                    total_loss_val += loss_val.item()
                    total_samples_val += targets_val.size(0)
                    total_correct_val += (predicted_val == targets_val).sum().item()

            val_loss = total_loss_val / len(validationloader)
            val_accuracy = total_correct_val / total_samples_val
            val_losses_epochs.append(val_loss)
            val_accuracy_epochs.append(val_accuracy)

            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        results[model_name]['train_losses'].append(train_losses_epochs)
        results[model_name]['val_losses'].append(val_losses_epochs)
        results[model_name]['train_accuracies'].append(train_accuracy_epochs)
        results[model_name]['val_accuracies'].append(val_accuracy_epochs)

# Average across the folds for plotting
for model_name in models.keys():
    results[model_name]['train_losses'] = np.array(results[model_name]['train_losses'])
    results[model_name]['train_losses_averaged'] = np.mean(results[model_name]['train_losses'], axis=0)
    results[model_name]['val_losses'] = np.array(results[model_name]['val_losses'])
    results[model_name]['val_losses_averaged'] = np.mean(results[model_name]['val_losses'], axis=0)
    results[model_name]['train_accuracies'] = np.array(results[model_name]['train_accuracies'])
    results[model_name]['train_accuracies_averaged'] = np.mean(results[model_name]['train_accuracies'], axis=0)
    results[model_name]['val_accuracies'] = np.array(results[model_name]['val_accuracies'])
    results[model_name]['val_accuracies_averaged'] = np.mean(results[model_name]['val_accuracies'], axis=0)

'''
Test phase
'''
testloader = torch.utils.data.DataLoader(data_blood_test, batch_size=BATCH_SIZE, shuffle=False)

# Evaluate on Test Set
for model_name, model in models.items():
    print(f'Evaluating {model_name}...')
    model.eval()
    total_loss_test = 0.0
    total_samples_test = 0
    total_correct_test = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs_test, targets_test in testloader:
            outputs_test = model(inputs_test)
            outputs_test = outputs_test.squeeze()
            targets_test = targets_test.squeeze().type(torch.long)
            _, predicted_test = torch.max(outputs_test, 1)

            # Calculate loss
            loss_test = loss_function(outputs_test, targets_test)
            total_loss_test += loss_test.item()

            total_correct_test += (predicted_test == targets_test).sum().item()
            total_samples_test += targets_test.size(0)

            all_predictions.extend(predicted_test.cpu().numpy())
            all_targets.extend(targets_test.cpu().numpy())

    test_loss = total_loss_test / len(testloader)
    test_accuracy = total_correct_test / total_samples_test

    print(f'Test Loss for {model_name}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Convert predictions and targets to numpy arrays for confusion matrix
    all_predictions_np = np.array(all_predictions)
    all_targets_np = np.array(all_targets)

    # Calculate confusion matrix
    class_labels = [str(i) for i in range(n_classes)]
    cm = confusion_matrix(all_targets_np, all_predictions_np)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    if model_name == 'CNN':
        plt.title('CNN Confusion Matrix')
    else:
        plt.title('Baseline Confusion Matrix')
    plt.savefig(f'plots/confusion_matrix_{model_name}.png')
    plt.show()

    # Calculate F1-score, recall, and precision
    f1 = f1_score(all_targets_np, all_predictions_np, average='weighted')
    recall = recall_score(all_targets_np, all_predictions_np, average='weighted')
    precision = precision_score(all_targets_np, all_predictions_np, average='weighted')
    accuracy = accuracy_score(all_targets_np, all_predictions_np)

    print(f'Average F1-score for {model_name}: {f1:.4f}')
    print(f'Average Recall for {model_name}: {recall:.4f}')
    print(f'Average Precision for {model_name}: {precision:.4f}')
    print(f'Average Accuracy for {model_name}: {accuracy:.4f}')
'''
Plot results
'''
epochs = np.arange(1, num_epochs + 1)

for model_name in models.keys():
    train_losses_folds_averaged = results[model_name]['train_losses_averaged']
    val_losses_folds_averaged = results[model_name]['val_losses_averaged']
    train_accuracy_folds_averaged = results[model_name]['train_accuracies_averaged']
    val_accuracy_folds_averaged = results[model_name]['val_accuracies_averaged']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_folds_averaged, 'blue', label='Training loss', marker = 'o')
    plt.plot(epochs, val_losses_folds_averaged, 'orange', label='Validation loss', marker = 's')
    plt.title(f'Average Training and Validation loss across folds for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_folds_averaged, 'blue', label='Training accuracy', marker = 'o')
    plt.plot(epochs, val_accuracy_folds_averaged, 'orange', label='Validation accuracy', marker = 's')
    plt.title(f'Average Training and Validation accuracy across folds for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/average_loss_accuracy_{model_name}.png')
    plt.show()

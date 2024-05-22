import os
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset
import medmnist
import seaborn as sns
from medmnist import INFO
from PIL import Image
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class baseLine(nn.Module):
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
dataset = ConcatDataset([data_blood_train, data_blood_val])

k_folds = 5
num_epochs = 5
loss_function = nn.CrossEntropyLoss() 

# k-fold cross validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# fold results
results = {}

# fixed number seed
torch.manual_seed(42)

#track losses
train_losses_folds = []
val_losses_folds = []
train_accuracy_folds= []
val_accuracy_folds = []
'''
Training phase
'''
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

    print(f'FOLD {fold+1}')
    print('--------------------------------')

    # select 
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    validation_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=BATCH_SIZE, sampler=train_subsampler)
    validationloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=BATCH_SIZE, sampler=validation_subsampler)
    # Print the first 5 batches of data from the training loader
    # init neural network
    model = baseLine()
    model.apply(reset_weights)


    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses_epochs= []
    val_losses_epochs = []
    train_accuracy_epochs= []
    val_accuracy_epochs = []
   # Run the training loop for defined number of epochs
    for epoch in range(num_epochs):
      print(f'Starting epoch {epoch+1}')

      # Training phase
      model.train()
      total_loss = 0.0
      total_samples_train = 0
      total_correct_train = 0

      for i, data in enumerate(trainloader, 0):
          inputs, targets = data
          #print("Sample training targets (labels):", targets[:10])
          optimizer.zero_grad()
          
          outputs = model(inputs)
          #print("Sample training outputs (labels):", outputs[:10])
          outputs = outputs.squeeze()
          #print("Sample training outpiuts:", outputs[:10])
          targets = targets.squeeze()
          #print("Sample training targets (labels):", targets[:10])
          targets = targets.type(torch.long)

          loss = loss_function(outputs, targets)
          loss.backward()
          optimizer.step()

          _, predicted = torch.max(outputs, 1)
          #print("Model training predictions:", predicted[:10])
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
              targets_val = targets_val.squeeze()
              targets_val = targets_val.type(torch.long)
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

    train_losses_folds.append(train_losses_epochs)
    train_accuracy_folds.append(train_accuracy_epochs)
    val_losses_folds.append(val_losses_epochs)
    val_accuracy_folds.append(val_accuracy_epochs)


#Average across the folds for plotting
train_losses_folds = np.array(train_losses_folds)
train_losses_folds_averaged = np.mean(train_losses_folds, axis=0)
train_accuracy_folds = np.array(train_accuracy_folds)
train_accuracy_folds_averaged = np.mean(train_accuracy_folds, axis= 0)

val_losses_folds = np.array(val_losses_folds)
val_losses_folds_averaged = np.mean(val_losses_folds, axis=0) 
val_accuracy_folds = np.array(val_accuracy_folds)
val_accuracy_folds_averaged = np.mean(val_accuracy_folds, axis= 0)

'''
Test phase
'''
testloader = torch.utils.data.DataLoader(
                data_blood_test,
                batch_size=BATCH_SIZE,
                shuffle=False  # Do not shuffle the test data
            )

# Initialize empty lists to store predictions and targets
all_predictions = []
all_targets = []

# Evaluate on Test Set
model.eval()
total_loss_test = 0.0
total_samples_test = 0
total_correct_test = 0

with torch.no_grad():
    for inputs_test, targets_test in testloader:
        
        outputs_test = model(inputs_test)
        outputs_test = outputs.squeeze()
        targets_test = targets.squeeze()
        targets = targets.type(torch.long)
        _, predicted_test = torch.max(outputs_test, 1)

        # Calculate loss    
        loss_test = loss_function(outputs_test, targets_test)
        total_loss_test += loss_test.item()

        total_correct_test += (predicted_test == targets_test).sum().item()
        total_samples_test += targets_test.size(0)

        # Append predictions and targets to the lists
        all_predictions.extend(predicted_test.cpu().numpy())
        all_targets.extend(targets_test.cpu().numpy())
        

# Calculate test loss
test_loss = total_loss_test / len(testloader)
# Calculate test accuracy
test_accuracy = total_correct_test / total_samples_test

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Convert predictions and targets to numpy arrays for confusion matrix
all_predictions_np = np.array(all_predictions)
all_targets_np = np.array(all_targets)


'''
Plotting
'''
# Calculate confusion matrix
class_labels = list(info['label'].values())
cm = confusion_matrix(all_targets_np, all_predictions_np)
print("Confusion Matrix:")
print(cm)
# Plot confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# Tilt the labels
plt.xticks(rotation=10, ha='right')
# Reduce font size
plt.tick_params(axis='y', which='both', labelsize=7) 
plt.tick_params(axis='x', which='both', labelsize=6) 
plt.title('Baseline - Confusion Matrix')
plt.savefig("plots/baseline_confusion_matrix.png")
plt.show()

# Plot the training and validation losses across epochs for each fold
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses_folds_averaged, label='Train', color='blue', linestyle='-', marker='o')
plt.plot(epochs, val_losses_folds_averaged, label='Validation', color='orange', linestyle='--', marker='s')
plt.title('Baseline - Training and Validation Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True)
plt.legend(loc='upper right', fontsize=12)
plt.savefig("plots/baseline_loss_plot.png")
plt.show()

# Plot the training and validation accuracy across epochs for each fold
plt.plot(epochs, train_accuracy_folds_averaged, label='Train', color='blue', linestyle='-', marker='o')
plt.plot(epochs, val_accuracy_folds_averaged, label='Validation', color='orange', linestyle='--', marker='s')
plt.title('Baseline - Training and Validation Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
#plt.axhline(x, label = "Chance") x-> find random prediction accuracy (e.g. 0.125 for balanced 8 classes) - creates horizontal line
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)
plt.legend(loc='upper left', fontsize=12)
plt.savefig("plots/baseline_accuracy_plot.png")
plt.show()




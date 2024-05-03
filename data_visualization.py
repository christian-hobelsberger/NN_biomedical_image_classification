import os
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_flag = "bloodmnist"
download = True
BATCH_SIZE = 128

info = INFO[data_flag]
task = info["task"]
n_channels = info["n_channels"]
n_classes = len(info["label"])

DataClass = getattr(medmnist, info["python_class"])

data_blood_train = torch.load("data/data_blood_train.pt")
data_blood_test = torch.load("data/data_blood_test.pt")
data_blood_val = torch.load("data/data_blood_val.pt")

# Visualizations

# 1. Visualize the 400 samples of the training set
montage_train_1 = data_blood_train.montage(length=1)
montage_train_20 = data_blood_train.montage(length=20)
montage_train_40 = data_blood_train.montage(length=40)

plots_folder = "plots/"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

montage_train_1.save(os.path.join(plots_folder, "montage_train_1.png"))
montage_train_20.save(os.path.join(plots_folder, "montage_train_20.png"))
montage_train_40.save(os.path.join(plots_folder, "montage_train_40.png"))

# 2. Visualize the distribution of the labels of the training set
df_labels = pd.DataFrame(data_blood_train.labels, columns=["Label"])

# Mapping of labels to their full names
label_names = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature Granulocytes",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet"
}

# Replace label codes with their full names
df_labels["Label"] = df_labels["Label"].map(label_names)

# Calculate the count of each label category and sort by index
label_counts = df_labels["Label"].value_counts().sort_index()

# Create a DataFrame directly from label counts
df_label_counts = pd.DataFrame({"Label": label_counts.index, "Count": label_counts.values})

# Display the table of label counts
print("Table of Label Counts:")
print(df_label_counts)

# Plot the distribution of labels
plt.figure(figsize=(16, 8))
sns.countplot(y='Label', data=df_labels, palette='pastel')
plt.title('Distribution of Labels in Training Set', fontsize=20)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Label', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig("plots/train_label_distribution.png")
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
plt.figure(figsize=(12, 6))
sns.countplot(y='Label', data=df_labels, palette='pastel')
# plt.title('Distribution of Labels in Training Set', fontsize=20)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Cell Type', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig("plots/train_label_distribution.png")

# 3. Visualize the distribution of the labels of all splits

# Function to create DataFrame and label counts
def create_label_df(data, split_name):
    # Create DataFrame with labels
    df_labels = pd.DataFrame(data.labels, columns=["Label"])

    # Replace label codes with their full names
    df_labels["Label"] = df_labels["Label"].map(label_names)

    # Calculate the count of each label category and sort by index
    label_counts = df_labels["Label"].value_counts().sort_index()

    # Create a DataFrame directly from label counts
    df_label_counts = pd.DataFrame({"Label": label_counts.index, "Count": label_counts.values})

    # Add split information
    df_label_counts['Split'] = split_name
    
    return df_label_counts

# Create label counts DataFrame for training set
df_train_label_counts = create_label_df(data_blood_train, 'Train')

# Create label counts DataFrame for validation set
df_val_label_counts = create_label_df(data_blood_val, 'Validation')

# Create label counts DataFrame for test set
df_test_label_counts = create_label_df(data_blood_test, 'Test')

# Concatenate label counts DataFrames
df_label_counts = pd.concat([df_train_label_counts, df_val_label_counts, df_test_label_counts])

# Reset index
df_label_counts.reset_index(drop=True, inplace=True)


# Calculate total count for each split
total_count_train = df_label_counts[df_label_counts['Split'] == 'Train']['Count'].sum()
total_count_val = df_label_counts[df_label_counts['Split'] == 'Validation']['Count'].sum()
total_count_test = df_label_counts[df_label_counts['Split'] == 'Test']['Count'].sum()

# Calculate relative frequencies for each split
df_label_counts['Relative Frequency'] = df_label_counts.apply(lambda row: row['Count'] / total_count_train if row['Split'] == 'Train' else (row['Count'] / total_count_val if row['Split'] == 'Validation' else row['Count'] / total_count_test), axis=1)

# Set the figure size
plt.figure(figsize=(12, 6))

# Create a shifted barplot with axis flipped
sns.barplot(x='Relative Frequency', y='Label', hue='Split', data=df_label_counts, palette='pastel')

# Set the title and labels
plt.title('Relative Distribution of Cell Types by Split', fontsize=16)
plt.xlabel('Relative Frequency', fontsize=14)
plt.ylabel('Cell Type', fontsize=14)

# Add a legend
plt.legend(title='Split', fontsize=12)

# Show plot
plt.tight_layout()
plt.savefig("plots/split_label_distribution.png")
plt.show()


# 4. Visualize the first three images for each label category
images_array = data_blood_train.imgs
labels_array = data_blood_train.labels.flatten()

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=len(label_names), figsize=(16, 6))

# Loop through each category and corresponding images
for i, label in enumerate(label_names.keys()):
    label_images = images_array[labels_array == label][:3]  # Select first three images for the category
    for j, image in enumerate(label_images):
        # Plot each image
        axes[j, i].imshow(np.array(image))
        axes[j, i].set_title(f'{label_names[label]}')
        axes[j, i].axis('off')

plt.tight_layout()
plt.savefig("plots/cell_type_examples.png")


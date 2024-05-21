import os
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from medmnist import INFO
import numpy as np
import pandas as pd
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import EarlyStopping, EpochScoring
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display
import seaborn as sns




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
          nn.Linear(64, 8),
          nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

def create_results_plot(df):
    sns.set_style(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=["#f2f2f2"]*df.shape[1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('grey')
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        else:
            cell.set_facecolor('#f2f2f2')

    plt.savefig("nice_table.png", bbox_inches='tight', dpi=300)
    plt.show()



data_blood_train = torch.load("data/data_blood_train.pt")
data_blood_test = torch.load("data/data_blood_test.pt")
data_blood_val = torch.load("data/data_blood_val.pt")

# Combine train and validation datasets
dataset = ConcatDataset([data_blood_train, data_blood_val])


# split into inputs and targets
def split_dataset(dataset):
    features = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]
    
    features = torch.stack(features)
    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.long).squeeze()
    return features, labels
    

# Split the combined dataset
X_train, y_train = split_dataset(dataset)


# print(f"Features shape: {X_train.shape}")
# print(f"Labels shape: {y_train.shape}")


# patience: Number of epochs to wait for improvement of the monitor value
# threshold: Ignore smaller improvements that the threshold value
# threshold_mode: Threshold value is interpeted as a relative value
early_stopping = EarlyStopping(patience=5, threshold=0.01, threshold_mode='rel', lower_is_better=True)


# Define the model
model = NeuralNetClassifier(
    convNet,
    criterion=nn.CrossEntropyLoss,
    max_epochs=5,
    optimizer=torch.optim.Adam,   
    verbose=0,
    callbacks=[early_stopping] 
)


# Define hyperparameter grid
# param_grid = {
#     'batch_size': [64, 128],
#     'max_epochs': [30, 50],
#     'lr': [0.001, 0.01, 0.1],
# }

# FOR TESTING
param_grid = {
    'batch_size': [64, 128],
    'lr': [0.001]

}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2, refit=False,verbose=3)
grid_result = grid.fit(X_train, y_train)

print(grid_result.cv_results_)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']



tune_result_df = pd.concat([pd.DataFrame(grid.cv_results_['params']), 
                            pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['validation_accuracy'])], axis=1)



create_results_plot(tune_result_df)



# Save the CV results to a CSV file
# cv_df = pd.DataFrame(grid.cv_results_)
# cv_df.to_csv('cv_results.csv', index=False)

# # Visualize the tuning results with parallel coordinate plot
# fig = px.parallel_coordinates(tune_result_df, color='mean_test_score', labels={'mean_test_score': 'Mean Test Score'})
# fig.show()


# cv_df = pd.DataFrame(grid.cv_results_)
# cv_df.to_csv('cv_results.csv', index=False)

# # Visualize the tuning results with parallel coordinate plot
# fig = px.parallel_coordinates(tune_result_df, color='mean_test_score', labels={'mean_test_score': 'Mean Test Score'})
# fig.show()



    

# Optimizing Blood Cell Classification with Convolutional Neural Networks

## Abstract

Biomedical image analysis plays a critical role in the diagnosis and treatment of patients with serious diseases. Advances in data availability and machine learning have improved the accuracy and efficiency of these analyses. This paper explores the use of convolutional neural networks (CNNs) for blood cell classification, a task essential for complete blood counts that identify infections based on blood cell characteristics. Manual assessment of blood cells is error-prone and resource-intensive, highlighting the need for automated solutions. We implemented a CNN using the BloodMNIST dataset, which contains over 17,000 images across eight blood cell classes. The CNN achieved superior performance with an accuracy of 0.914, significantly outperforming the baseline 1-layer Feedforward Neural Network on all evaluated metrics. Our results demonstrate that CNNs, with their ability to capture spatial hierarchies, provide a robust solution for blood cell classification, offering improved accuracy and reliability for medical diagnostics.

## Results
### Multi-label with Spans

| Metric  | CNN Model    | Baseline |
|--------|----------------------|---------------|
| Accuracy average | 0.914    | 0.781         |
| Precision average        | 0.912     | 0.775         |
| Recall average       | 0.896 | 0.738         |
| F1 Score average       | 0.901 | 0.748         |

## Usage instructions

Clone the repository

```
git clone https://github.com/christian-hobelsberger/NN_biomedical_image_classification.git
```

### Repository Structure

This repository is organized into several key directories and files, each serving a distinct purpose in the project workflow:

- [.config](https://github.com/christian-hobelsberger/NN_biomedical_image_classification/tree/main/.config): Configuration files for project settings.
- [data_visualization.py](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/blob/main/data_visualization.py): Python script for data visualization.
- [plots](https://github.com/christian-hobelsberger/NN_biomedical_image_classification/tree/main/plots): Directory containing plots generated from the project.
- [model_baseline.py](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/blob/main/model_baseline.py): Python script for the baseline model with fixes and improved plots.
- [model_cnn.py](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/blob/main/model_cnn.py): Python script for the CNN model with fixes and improved plots.
- [model_parameter_tuning.py](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/blob/main/model_parameter_tuning.py): Python script for model parameter tuning with minor fixes.
- [models.py](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/blob/main/models.py): Python script containing code for the deployed models.

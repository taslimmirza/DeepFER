# DeepFER: Facial Emotion Recognition Using Deep Learning

## Introduction

DeepFER: Facial Emotion Recognition Using Deep Learning aims to leverage the power of Convolutional Neural Networks (CNNs) and Transfer Learning to build a robust and efficient facial emotion recognition system. By training the model on a diverse dataset and employing advanced techniques, the project seeks to achieve high accuracy and real-time processing capabilities, enabling its use in real-world scenarios.

## Problem Statement

### Challenge 1

Facial emotion recognition systems face the challenge of accurately and efficiently identifying and classifying human emotions from facial expressions.

### Challenge 2

The inherent variability in facial expressions, subtle differences between emotions, and the influence of individual and cultural factors make it difficult to develop a universally accurate and reliable system.

## Implementation

### 1. Dataset Preprocessing

### 2. Dataset Augmentation

### 3. Model Training

### 4. Model Evaluation

### 5. Inference

## Dataset

The FER dataset contains 35,887 images of human faces, labeled with seven different facial expressions: angry, disgust, fear, happy, sad, surprise, and neutral. The dataset is split into two subsets, a training set of 28,821 images and a test set of 7,066 images. The images are grayscale, 48×48 pixels in size, and the data is stored in jpg format.

### Samples Occurrence

The number of images per class is illustrated in the table below and plotted in the next slide. We can observe that the dataset is highly imbalanced.

| Class     | Angry | Disgust | Fear | Happy | Neutral | Sad  | Surprise |
|-----------|-------|---------|------|-------|---------|------|----------|
| **Train** | 3993  | 436     | 4103 | 7164  | 4982    | 4938 | 3205     |
| **Test**  | 960   | 111     | 1018 | 1825  | 1216    | 1139 | 797      |

### Class Weights Calculation

The imbalanced dataset creates challenges for the model to generalize across all classes. To address this issue, we calculate each class's weight and feed that dictionary to the model.

\[ \text{weight}_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_{\text{samples}_j}} \]

- \( \text{weight}_j \) is the weight for class \( j \)
- \( n_{\text{samples}} \) is the total number of samples (rows) in the dataset
- \( n_{\text{classes}} \) is the total number of unique classes in the target variable
- \( n_{\text{samples}_j} \) is the number of samples that belong to class \( j \)

### Dataset Augmentation

As we are using transfer learning, which sometimes leads to overfitting due to model complexity, we augment our data to avoid overfitting during training. Typically, pixel values in images range between 0 and 255. Normalizing them to values between 0 and 1 often improves model training.

We use the following augmentations to transform the dataset:
- **Rotation**: Rotate images up to 10 degrees.
- **Width Shift**: Shift images horizontally by up to 10% of the width.
- **Height Shift**: Shift images vertically by up to 10% of the height.
- **Zoom**: Zoom in or out by up to 10%.
- **Horizontal Flip**: Randomly flip images horizontally.

## Model Development

For the backbone of our transfer learning model, we use the VGG16 pretrained model, previously trained on the ImageNet dataset. We develop a sequential model using a pre-trained base model followed by normalization, pooling, and flattening layers. Two fully connected layers with regularization and dropout, culminating in a softmax layer for 7-class classification.

## Model Training

The model is compiled with the Adam optimizer, categorical cross-entropy as the loss function, and various performance metrics including accuracy, precision, recall, AUC, and a custom F1 score. The model training process includes callbacks to reduce the learning rate on plateau, save the best model, and stop early if no improvement is observed. The model is fit on training and validation datasets for 100 epochs with class weights applied to address class imbalance. Other parameters include a batch size of 64 and a learning rate of 0.0001.

## Evaluation

### Training and Validation Results

The "Train" results show how well the model performed on the data it was directly trained on, while the "Validate" results provide a more realistic estimate of how the model will perform in the real world on unseen data.

| Metric Name | Train | Validate |
|-------------|-------|----------|
| Loss        | 0.603 | 1.204    |
| Accuracy    | 0.949 | 0.903    |
| Precision   | 0.870 | 0.692    |
| Recall      | 0.757 | 0.579    |
| AUC         | 0.976 | 0.908    |
| F1_Score    | 0.809 | 0.630    |

### Training and Validation Graphs

The graphs illustrate the training and validation metrics over the epochs, helping to understand how the model performance evolves during training.

### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's classification performance, showing the true vs. predicted labels for the validation dataset. The model appears to have decent accuracy but struggles with distinguishing between certain emotions.

## Inference

The trained model is used to infer the emotion of a person in a test image. This involves loading the image, preprocessing it, passing it through the model, and displaying the predicted emotion.

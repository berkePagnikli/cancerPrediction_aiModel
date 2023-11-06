# Cancer Detection Model using TensorFlow and Keras

## Overview
This project includes a machine learning model to detect cancer using TensorFlow and Keras. It is designed as a simple artificial intelligence model built with the purpose of learning and improving machine learning skills.

## Dataset
The model is trained on a dataset contained in `cancer.csv`, which includes various features used to predict cancer diagnosis (1 for malignant, 0 for benign).

## Features of the Model
- A sequential neural network built with TensorFlow.
- Batch normalization for stable and faster convergence.
- Leaky ReLU activation functions to address the dying ReLU problem.
- L2 regularization to mitigate overfitting.
- Early stopping and adaptive learning rates for optimal training performance.

## Installation
Before running this project, ensure you have the following packages installed:

- pandas
- scikit-learn
- TensorFlow
- Keras
- matplotlib

## Usage

To train and evaluate the model, follow these steps using the script provided in the repository:

1. **Load the dataset:**
   - The dataset should be in CSV format and named `cancer.csv`.

2. **Prepare the data:**
   - The script will split the data into training and testing sets.

3. **Define the neural network architecture:**
   - The architecture is defined within the script using TensorFlow and Keras.

4. **Compile the model:**
   - The model is compiled with Adam optimizer and binary cross-entropy loss.

5. **Train the model:**
   - Training is executed with early stopping and learning rate reduction on plateau.

6. **Evaluate the model:**
   - After training, the model's performance is evaluated on the test set.

7. **Save the model:**
   - The trained model is saved to a file named `cancerPredictor.h5`.

8. **Plot the training and validation loss curves:**
   - The script will generate plots to visualize the loss during the training process.

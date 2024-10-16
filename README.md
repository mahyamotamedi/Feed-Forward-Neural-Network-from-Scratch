# Feed-Forward Neural Network from Scratch

## Overview
This project implements a basic feed-forward neural network from scratch using Python. It was designed to classify images based on pixel intensity values from a given dataset. This project represents my first interaction with neural networks, and the goal was to learn the fundamental mechanics of forward and backward propagation.

## Datasets
The datasets used for this project include:
- **Training images**: Stored in `train_images.csv`
- **Training labels**: Stored in `train_labels.csv`
- **Test images**: Stored in `test_images.csv`
- **Test labels**: Stored in `test_labels.csv`

The images are grayscale, with pixel values ranging between 0 and 255.

## Implementation Details

### 1. **Dataloader Class**
- **Task**: Loads the dataset, manages data batching, shuffles the data, and one-hot encodes the labels. It enables efficient data handling during training.

### 2. **Activation Function Classes**
- **Identical Class**: Applies a linear activation function, typically used in layers where no transformation is required (e.g., output layers for regression).
- **ReLU Class**: Introduces non-linearity by zeroing out negative values and keeping positive ones, commonly used in hidden layers.
- **LeakyReLU Class**: A variation of ReLU that allows a small, non-zero gradient for negative inputs to prevent neuron death.
- **Softmax Class**: Converts raw output scores into probabilities that sum up to 1, typically used in the output layer for classification tasks.
- **Sigmoid Class**: Squashes input values into a range between 0 and 1, commonly used in binary classification or output layers.
- **Tanh Class**: Similar to sigmoid but squashes input values between -1 and 1, often used in hidden layers for more balanced output values.

### 3. **CrossEntropy Class**
- **Task**: Calculates the cross-entropy loss, a common choice for classification tasks. It measures the difference between predicted probabilities and actual labels.

### 4. **Layer Class**
- **Task**: Represents an individual layer in the network, holding weights, biases, and the activation function. Manages the forward and backward passes during training.

### 5. **FeedForwardNN Class**
- **Task**: The core class that builds the neural network. It allows adding layers, defining activation functions, and setting up the loss function and optimizer. Manages the entire training process, including weight updates and performance tracking.

### Training and Evaluation
- The network consists of multiple layers added sequentially. Each layer applies its activation function to the data and passes it to the next layer.
- The **CrossEntropy** loss function evaluates the performance, and backpropagation is used to adjust the weights of the layers.

course: Artificial Intelligence-University of Tehran

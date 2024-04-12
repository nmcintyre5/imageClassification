# Image Classification with TensorFlow

## Overview
This script demonstrates basic image classification with TensorFlow and Keras. The script creates, trains, and evaluates a neural network model is able to predict digits from hand-written digits with a high degree of accuracy.

## Key Features
- Loads the MNIST dataset [mnist.npz](/mnist.npz)
- Preprocesses the data including one-hot encoding and normalization
- Constructs a neural network model, utilizing TensorFlow
![screenshot of model summary](/Run_images/model_summary.png)
- Trains the model on the training data
- Evaluates the model's performance on the test set
![predictions](/Run_images/Predictions.png)
- Visualizes predictions with sample images and prediction probabilities
- Generates a confusion matrix to evaluate classification performance
- Plots training and validation loss and accuracy 
![loss](/Run_images/Loss.png)
![accuracy](/Run_images/accuracy.png)

## Credits
This project was adapted from a Coursera course Basic Image Classification with TensorFlow by Amit Yadav.
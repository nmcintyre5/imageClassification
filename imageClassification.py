import logging
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set TensorFlow logging verbosity to suppress info messages
tf.get_logger().setLevel(logging.ERROR)
print('Using TensorFlow version', tf.__version__)

# Load MNIST dataset from .npz file
with np.load('mnist.npz') as data:
    # Extracting training and testing data
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

# One-hot encode the target labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Reshape training and testing data to 2D arrays
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

# Feature scaling using z-score normalization
# Calculate mean and standard deviation of the training data
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

# Define a small constant to avoid division by zero
epsilon = 1e-10

# Normalizes the training and testing data
x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

# Creating the model architecture
model = Sequential([
    Input(shape=(784,)),  # Input layer with 784 neurons
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(128, activation='relu'),  # Second hidden layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax'),  # Output layer with 10 neurons (one for each class) and softmax activation
])

# Compile the model
model.compile(
    optimizer='sgd',  # Stochastic Gradient Descent optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Evaluation metric
)

# Print model summary
model.summary()

# Train the model
model.fit(x_train_norm, y_train_encoded, epochs=3)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set model accuracy: ', accuracy * 100, "%")

# Make predictions on the test set
predictions = model.predict(x_test_norm)
print('Shape of predictions: ', predictions.shape)

# Visualize predictions
plt.figure(figsize=(12, 12))
start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    prediction = np.argmax(predictions[start_index + i])
    ground_truth = y_test[start_index + i]

    col = 'g'
    if prediction != ground_truth:
        col = 'r'

    plt.xlabel('predicted={}, true={}'.format(prediction, ground_truth), color=col)
    plt.imshow(x_test[start_index + i], cmap='binary')

plt.show()

# Plot the prediction probabilities for a specific sample
plt.plot(predictions[8])
plt.show()
# Generate predictions on test data
predictions = model.predict(x_test_norm)

# Convert predictions to class labels
y_pred = np.argmax(predictions, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Define number of epochs
epochs = 10

# Train the model and store training history
history = model.fit(x_train_norm, y_train_encoded, epochs=epochs, validation_data=(x_train_norm, y_train_encoded))

# Retrieve loss and accuracy from training history
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs_range = range(1, epochs + 1)

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()